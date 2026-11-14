# Full system: funzionamento e risultati dell’ultima run

Run: `full_system_20261114_111550_9145ef8f`, avviata il **14 novembre 2026 alle 11:15:50**. Il movimento termina alle **11:18:29**, coerentemente con il video `llm+sam3+graspgen.webm`, modificato alle 11:18:31.

**Esito:** Bone Forceps trasferito dal vassoio 2 al vassoio 1, con movimento e deposito verificati nella simulazione. La presa usa un vincolo fisso ideale: forza, attrito e stabilità fisica non sono validati.

## 1. Controlli iniziali

Il launcher verifica dipendenze, mesh della scena, pinza, Ollama, pesi SAM3 e server GraspGen. In questa run tutti i controlli sono superati. Il profilo completo richiede realmente Ollama e SAM3: gli errori interrompono la pipeline.

## 2. LLM: dalla frase al task

Ollama esegue `qwen2.5:3b` e interpreta:

> Pick up the Bone Forceps from tray 2 and place it in tray 1.

Il risultato validato è: azione `pick_and_place`, strumento `Bone Forceps`, parola per la segmentazione `forceps`, sorgente `tray_2`, destinazione `tray_1`. Durata dello stadio: **7,511 s**. Il LLM definisce il compito; le pose della pinza vengono calcolate dopo.

Fonte: `plan.json`.

## 3. Acquisizione e SAM3: individuare lo strumento

PyBullet crea la scena con un solo strumento e acquisisce immagine RGB e profondità a **640 × 480 pixel**. SAM3 gira su CPU e riceve il prompt `forceps`. Viene scelta la maschera dell’istanza con score massimo.

Risultati: **score 0,8895**, **928 pixel** nella maschera, bounding box `[364, 178, 461, 264]`. Durata SAM3: **31,954 s**. La maschera usata proviene da SAM3; la maschera ground truth salvata dal simulatore serve come riferimento.

Fonti: `sim_capture.json`, `sim_segmentation.json`.

## 4. Dalla maschera alla nuvola 3D

Per ciascun pixel della maschera, la profondità metrica Z e i parametri della camera permettono di ricostruire:

```text
X = (u − cx) × Z / fx
Y = (v − cy) × Z / fy
p_base = T_base_camera × [X, Y, Z, 1]ᵀ
```

La trasformazione camera → base robot porta tutti i punti nello stesso riferimento del robot, `lbr_link_0`.

Risultati: **928 punti**, centroide in base robot **[0,572964; −0,257810; 0,099460] m**. Lo scarto del centroide dalla posizione di riferimento del simulatore è **2,78 mm**: è un confronto diagnostico, non una correzione del centroide. Durata: **0,597 s**.

Fonte: `sim_deprojection.json`.

## 5. GraspGen: generare e ordinare i candidati

Il client sottrae il centroide alla nuvola (`p_centrato = p_base − centroide`) e invia i punti al server GraspGen. Il modello **diffusion** genera possibili pose della pinza Robotiq 2F-140; il **discriminator** assegna a ogni posa uno score. Il client riordina globalmente i candidati per score decrescente.

Un candidato contiene posizione e orientamento della pinza, espressi tramite una matrice 4 × 4. Lo score misura la confidenza del modello; non certifica raggiungibilità, assenza di collisioni o successo fisico.

Risultati: **200 candidati restituiti**, durata dello stadio **1,275 s**. Il migliore per score è `server_index=0`, con **0,934123**. Questo candidato viene poi scartato perché la geometria di chiusura collide con il tavolo.

Fonte: `grasp_graspgen.json`. L’anteprima `grasp_graspgen_overview.png` mostra al massimo 20 candidati, mentre il JSON li conserva tutti.

## 6. Trasformazioni: dalla posa generata al TCP

La notazione `T_A_B` indica la posa del riferimento B espressa nel riferimento A. La matrice contiene rotazione R e traslazione t:

```text
T = [ R  t ]
    [ 0  1 ]

T_base_gripper = T_base_centered × T_centered_gripper
T_base_tcp     = T_base_gripper × T_gripper_tcp
T_world_tcp    = T_world_base × T_base_tcp
```

- `T_base_centered` restituisce il centroide sottratto prima dell’inferenza.
- `gripper` è l’origine della pinza prevista da GraspGen; `TCP` è il punto operativo usato per il movimento.
- `T_gripper_tcp` applica un offset di **0,195 m lungo Z locale** e una rotazione di **−90° attorno a Z**. L’offset va quindi ruotato con la pinza: non si somma semplicemente alla quota verticale del mondo.
- In questa scena `T_world_base` è identità: coordinate mondo e base robot coincidono.

Convenzione GraspGen: avvicinamento lungo **+Z locale**, chiusura lungo **+X locale** della pinza.

## 7. Selezione della presa eseguibile

Il planner esamina i candidati in ordine di score e verifica geometria delle dita, chiusura, cinematica inversa (IK), limiti articolari e collisioni campionate lungo il percorso.

Prima della scelta vengono scartati **42 candidati**:

| Motivo | Numero |
|---|---:|
| Geometria delle dita / percorso di chiusura | 34 |
| Corridoio presa → pre-presa | 2 |
| Trasferimento | 5 |
| IK della presa | 1 |

Viene scelto **`server_index=42`, 43° in graduatoria**, score originale **0,873540**. Il planner corregge la posa usando CAD e pose note della scena: rialzo parametrico **40 mm** e spostamento di **−5,21 mm** lungo l’asse di chiusura. La traslazione complessiva in base è **[−3,994; +3,343; +39,860] mm**, senza cambiare orientamento. La posa corretta non viene rivalutata dal modello: lo score resta quello originale.

Risultato operativo dopo la correzione:

| Grandezza | Valore |
|---|---|
| Posizione TCP in base/mondo | `[0,577021; −0,253303; 0,133453] m` |
| Orientamento TCP, quaternione xyzw | `[0,905308; 0,422121; −0,035341; 0,031352]` |
| Apertura finale stimata della pinza | **40,65 mm** |
| Tempo di pianificazione | **32,75 s** |

`selected` conserva il migliore per score; **`simulation_selected` contiene la presa corretta scelta per l’esecuzione**. Il centroide della nuvola, l’origine della pinza e il TCP sono punti diversi.

Fonti: `grasp_simulation_selection.json`, `graspgen_sim_motion_validation.json`.

## 8. Movimento e deposito

La pipeline usa IK con più inizializzazioni, pianificazione RRT-Connect e temporizzazione quintica. PyBullet esegue:

```text
Approach → Descend → Close → Lift → Transfer → Place → Open → Retreat → Return
```

Alla chiusura, lo strumento viene collegato alla pinza con un **vincolo fisso ideale**; all’apertura viene rilasciato.

| Risultato misurato | Valore |
|---|---|
| Stato del movimento | `COMPLETED` |
| Durata simulata | **79,54 s** |
| Errore TCP massimo a fine fase | **0,099 mm** |
| Errore orientamento massimo a fine fase | **0,009°** |
| Controlli di esecuzione e collisioni campionate | Superati |
| Posizione finale strumento nel mondo | `[0,598003; 0,415966; 0,096012] m` |
| Centro nel vassoio destinazione | Sì |
| Strumento assestato sulla destinazione | Sì, con **4 contatti** |

La verifica collisioni è campionata ogni **1/60 s** e riguarda la scena configurata; il report mantiene `collision_free_path_validated=false`, pur riportando `sampled_collision_checks_passed=true`. Non equivale a una garanzia continua del percorso.

Fonte: `sim_robot_motion.json`.

## 9. Come leggere lo stato finale

Al momento della lettura, `sim_robot_motion.json` e `graphical_run.json` riportano `COMPLETED`, mentre `run_manifest.json` riporta ancora `RUNNING`. Il report di pianificazione conserva inoltre `PLANNED_NOT_EXECUTED`: per sapere se il movimento è stato eseguito va letto **`sim_robot_motion.json`**.

La run dimostra l’uso di LLM, SAM3 e GraspGen e il deposito nella simulazione. La selezione geometrica sfrutta anche il CAD noto della scena. Non valida presa per attrito/forza, zone funzionali chirurgiche o trasferimento sul robot reale.

Tutti i JSON citati si trovano in:
`data/outputs/full_system_20261114_111550_9145ef8f/`.
