# Modello condiviso: KUKA LBR iiwa 7 R800

`config/robot.yaml` seleziona l'iiwa 7 R800 per scena PyBullet, validatore di collisioni,
FK/IK, copia cinematica del controller, controller interattivo e rendering offline.
Giunti e flangia vengono risolti per nome. Non ci sono sostituzioni automatiche con l'iiwa 14.

La fonte è la [descrizione ufficiale LBR Stack](https://github.com/lbr-stack/lbr_fri_ros2_stack/blob/52f816bd59cbd2390efc890cca0ad61389e4ccb3/lbr_description/urdf/iiwa7/iiwa7_description.xacro),
alla stessa revisione fissata in `ros2/workspace.repos`: `52f816bd59cbd2390efc890cca0ad61389e4ccb3`.
Il preparatore conserva assi, dimensioni, inerzie dei link fisici e limiti upstream.
Omette i tag ROS/Gazebo, usa gli STL di collisione ufficiali anche per il rendering
e assegna massa zero al frame virtuale di flangia. Sorgenti e licenza sono conservati.

## Preparazione dopo il pull

Con il runtime ROS già costruito, dalla radice del repository:

```bash
bash scripts/run_ros2_system.sh prepare
bash scripts/run_ros2_system.sh self-test
bash scripts/run_ros2_system.sh --gui \
  --instruction "Pick up the Bone Forceps from tray 2 and place it in tray 1."
```

Per il primo avvio seguire [full-system-ros2.md](full-system-ros2.md), inclusi servizi AI,
pesi e strumenti. Gli asset generati restano in `data/external/` e vengono ricreati dal
codice versionato: non serve trasferire archivi o modificare percorsi assoluti.

In un ambiente Python nativo, per il solo braccio:

```bash
python3 -m pip install -r requirements-core.txt
python3 -m pipeline.robot_assets
```

Il download richiede rete al primo utilizzo. Un checkout con revisione errata o modifiche
locali viene segnalato e lasciato intatto. Asset mancanti o hash incoerenti fermano il preflight.

## Giunti, flangia e TCP

| Elemento | Valore condiviso |
|---|---|
| Base | `lbr_link_0` |
| Giunti ROS e URDF | `lbr_A1` … `lbr_A7` |
| Flangia | `lbr_link_ee`, con giunto fisso a +35 mm da `lbr_link_7` |
| Limiti articolari | ±[170, 120, 170, 120, 170, 120, 175] gradi |
| Massime velocità URDF | [98, 98, 100, 130, 140, 180, 180] gradi/s |
| Massimo effort URDF | 200 Nm per giunto |
| Flangia → TCP demo | [0, 0, 0.100] m |
| Flangia → TCP Robotiq | [0, 0, 0.195] m |

I TCP sono definiti in `pipeline/sim/gripper_geometry.py`, con orientamento uguale alla
flangia; la rotazione di montaggio della mesh Robotiq rispetto alla convenzione GraspGen
è gestita separatamente dalla geometria della pinza. Il planner usa il TCP della pinza
selezionata in configurazione. Gli offset sono nominali: andranno misurati sul montaggio reale.

Il controller ricava le velocità dal medesimo URDF; un limite configurato può soltanto
restringerle. Restano i limiti operativi più lenti del planner. Accelerazione e jerk sono
parametri della simulazione, non limiti del costruttore ricavati dal modello.

Le nuove run registrano `robot_description` con sorgente, hash degli asset, nomi dei frame,
limiti e TCP. Il planner e il coordinatore controllano l'identità del modello tra cattura,
validazione ed esecuzione. Il profilo `pybullet_iiwa14` serve esplicitamente alle vecchie
registrazioni; le loro traiettorie articolari non vengono riutilizzate come traiettorie iiwa 7.

## Verifiche

`tests/test_robot_model.py` verifica i limiti upstream e confronta la FK di scena,
validatore e copia del controller con un prodotto indipendente delle trasformazioni URDF:
dieci pose, base traslata/ruotata ed entrambi i TCP. Verifica anche IK, rifiuto di mesh
alterate, assenza di fallback e rispetto delle velocità URDF. Con tutti i giunti a zero,
la flangia è a quota 1.266 m rispetto alla base (inclusi i 35 mm finali).

La prova ROS del 14 novembre 2026 in `data/outputs/ros2_iiwa7_validation_20261114_2/`
riusa proposte neurali registrate e ripianifica integralmente sul nuovo modello:
7 action braccio, 2 pinza e deposito verificato. La suite Linux/Humble completa passa
189 test. Queste prove verificano la coerenza della simulazione; il trasporto mantiene
il vincolo ideale già dichiarato e non certifica la tenuta fisica della pinza.
