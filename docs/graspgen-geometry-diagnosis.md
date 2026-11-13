# Diagnosi geometrica della run 20261113_123556

Fonte: `data/outputs/graspgen_pickplace_20261113_123556/`.
La run dell'utente ha restituito 200 prese, tutte rifiutate prima dell'IK:
155 per geometria pinza/ambiente, 44 per sezione nominale assente e una per
apertura stimata eccessiva. Tempo registrato: 30,379 s.

Una lettura statica delle mesh e della cinematica URDF, senza client fisico,
mostra per il candidato 81:

- piano superiore del vassoio: Z = 0,090 m;
- TCP nominale candidato: Z = 0,099826 m;
- polpastrello sinistro aperto: Z minimo = 0,080910 m;
- polpastrello sinistro alla larghezza stimata 0,041897 m:
  Z minimo = 0,058694 m.

Il riferimento TCP nominale non è l'estremità inferiore delle dita. Un TCP
vicino alla superficie dell'oggetto non garantisce che la Robotiq possa arrivarci
o chiudersi sopra il vassoio. Una diversa traiettoria del braccio non elimina
questa intersezione alla posa finale.

## Diagnostica preparata, da eseguire con autorizzazione

`scripts/diagnose_graspgen_geometry.py` legge solo la cattura e le pose già
salvate. Crea una cella CPU DIRECT per distanze e IK, senza `stepSimulation`,
senza avviare la GUI, senza server, inferenza o comandi al robot della scena.
Scrive un nuovo JSON e rifiuta di sovrascrivere un report esistente.

`pipeline/sim/grasp_refinement.py` cerca una correzione limitata a 40 mm verso
l'alto e 15 mm lungo l'asse di chiusura, senza modificare la rotazione. Richiede
prossimità bilaterale dei polpastrelli alla mesh, assenza di penetrazione e
controllo dell'intera chiusura. La larghezza deriva dalla geometria articolata,
non da una sezione approssimata attorno al TCP.

Una posa corretta è una proposta derivata, non l'output originale della rete:
il report conserva la posa originale, la correzione e le distanze. Il punteggio
originale non viene presentato come punteggio ricalcolato della posa modificata.
Questo non dimostra attrito, forza, stabilità o successo della presa fisica.

L'esistenza di una correzione ammissibile deve essere verificata dalla diagnostica;
non è garantita né ottenuta riducendo i margini o disattivando le collisioni.

## Esito della diagnostica eseguita dall'utente

`data/outputs/graspgen_geometry_diagnostic_20261113_130817.json` riporta il primo
superamento al candidato 30: correzione verticale 25 mm e lungo chiusura
-6,185 mm, apertura 30,971 mm, distanze dei due polpastrelli 0,322 e 0,202 mm.
L'IK da HOME risulta convergente: errore 0,0633 mm e margine articolare 0,1812 rad.
Non sono stati verificati da questo risultato percorso, tracking o trasporto.

La selezione di produzione usa ora la stessa funzione di raffinamento CAD.
`scripts/diagnose_graspgen_path.py` consente all'utente di controllare l'intera
traiettoria da HOME con lo stesso pianificatore, su cattura e candidati salvati.
Non chiama `tick`, non avanza la fisica, non rigenera la nuvola e non invia
richieste di inferenza. Gli input vengono copiati in una nuova cartella: i dati
originali sono conservati. `PATH_CHECK_PASSED_NOT_EXECUTED` significa soltanto
superamento dei controlli cinematici e geometrici campionati, non esecuzione.

## Esito della verifica percorso eseguita dall'utente (13:17)

La ricerca è terminata in 42,426 s senza esecuzione: 175 scarti geometrici,
4 nel corridoio pregrasp per distanza insufficiente fra link del braccio,
1 per IK e 20 all'inizio del sollevamento. Per questi 20 l'errore era il
vincolo erroneo di 0,5 mm fra strumento e vassoio anche al campione iniziale,
quando lo strumento è ancora appoggiato. Il candidato 30 appartiene invece
ai quattro rifiutati per distanza fra link; la sola IK alla presa era insufficiente.

La regola di supporto distingue ora il contatto dello strumento con la faccia
superiore del vassoio da una collisione. È ammesso solamente sul vassoio di
origine durante LIFT e su quello di destinazione durante PLACE/RELEASE, con
normale rivolta verso l'alto (dot >=0,95) e tolleranza numerica di 10 micrometri.
Margini del robot/pinza e controlli durante il trasferimento restano attivi.
La stessa regola si applica ai campioni del piano e al controllo misurato runtime.
La correzione richiede una nuova verifica manuale del percorso.

## Prima esecuzione GUI (run 20261113_143643)

Il percorso è stato ammesso per il candidato 112, score originale 0,81143.
La GUI ha eseguito avvicinamento, discesa e chiusura; il controllo runtime ha
fermato il sollevamento dopo 17,9625 s. Nessun trasferimento è stato completato.
La distanza strumento/supporto era -0,010009 mm a fronte di -0,010000 mm ammessi:
uno scarto di circa 9 nm, compatibile con rumore numerico al limite di confronto.

Il confronto del solo contatto di supporto esplicito ha ora epsilon 1e-7 m
(0,1 um), senza modificare i margini di braccio o pinza. La scena Robotiq usa
esplicitamente contactSlop=1e-6 m, inferiore alla tolleranza del controllore
di 1e-5 m. Il parametro è esposto nell'[API ufficiale PyBullet](https://github.com/bulletphysics/bullet3/blob/master/examples/pybullet/pybullet.c).
La GUI colora di rosso la selezione dopo un errore di esecuzione e il report
conserva fase e tempo dell'arresto. Queste modifiche richiedono una nuova
esecuzione manuale, senza dedurre il successo del trasporto dalla sola chiusura.
