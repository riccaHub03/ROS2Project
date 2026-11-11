# ROS 2

## Cosa c'è nel repository

| Percorso | Contenuto |
|---|---|
| `ros2/surgical_executor/` | Package `ament_python` del progetto: nodo `pick_and_place` che esegue il movimento pianificato dalla pipeline |
| `ros2/workspace.repos` | Repository ROS 2 esterni, bloccati alle revisioni usate (`llm_ros`, `sam3_ros`, `fri`, `lbr_fri_idl`, `lbr_fri_ros2_stack`) |
| `pipeline/motion/pick_and_place.py` | Logica pura della sequenza di pick & place (nessuna dipendenza da ROS): è testata in `tests/test_pick_and_place_plan.py` |
| `scripts/ros2_build.sh` | Build con colcon, da eseguire nel container |

La libreria Python `pipeline` resta indipendente da ROS: gli stadi si parlano con i report JSON. ROS 2 entra
solo nell'ultimo passo, l'esecuzione del movimento. Il nodo importa `pipeline` (nel container `/workspace` è
già nel `PYTHONPATH`, lo imposta `docker/entrypoint.sh`).

## Compilare

Nel container:

```bash
bash scripts/ros2_build.sh          # solo surgical_executor (veloce)
bash scripts/ros2_build.sh --all    # anche i repository di ros2/workspace.repos (richiede vcstool)
```

Poi apri un nuovo terminale nel container: l'entrypoint carica `ROS2_workspace/install`.

## Eseguire il pick & place

Con la pipeline già eseguita (esiste `data/outputs/grasp.json`) e la piattaforma di simulazione in funzione:

```bash
ros2 launch surgical_executor pick_and_place.launch.py destination_tray:=tray_1
# oppure, con parametri espliciti:
ros2 run surgical_executor pick_and_place --ros-args \
    -p grasp_report:=data/outputs/grasp.json -p destination_tray:=tray_2
```

Il nodo **rifiuta il movimento** se `grasp.json` non ha `readiness == VALIDATED_GEOMETRY_READY_FOR_TRAJECTORY_PLAN`
e non invia nessun comando. Quote, tempi e vassoi vengono da `config/surgical_environment.yaml`
(`motion_control`); topic e frame sono parametri in `ros2/surgical_executor/config/executor.yaml`.

| Parametro | Default | Significato |
|---|---|---|
| `grasp_report` | `data/outputs/grasp.json` | report di presa della pipeline |
| `config_path` | `config/surgical_environment.yaml` | configurazione dell'ambiente |
| `destination_tray` | `tray_1` | vassoio di deposito |
| `frame_id` | `lbr_link_0` | riferimento delle pose (base del robot) |
| `path_topic`, `status_topic`, `gripper_topic`, `ee_pose_topic`, `joint_states_topic` | `/lbr_tutorial/...`, `/lbr/...` | interfaccia verso la piattaforma |
| `render_video` | `true` | video offline della traiettoria registrata |

Al termine salva `data/outputs/robot_trajectory.json` e, se richiesto, il video.

## Limite attuale: robot reale

I topic di default sono quelli della piattaforma di simulazione `lbr_pybullet_tutorial` (percorso cartesiano,
stato, comando della pinza). **Il nodo non comanda ancora il KUKA reale.** Per l'hardware serve un nuovo
*backend* basato su `lbr_fri_ros2_stack` (ros2_control / MoveIt con `FollowJointTrajectory` e il driver della
pinza), con IK, collisioni e limiti dei giunti verificati lì. La sequenza di waypoint
(`pipeline/motion/pick_and_place.py`) e il blocco di sicurezza sulla presa si riusano così come sono: va
sostituito solo `send_path` in `executor_node.py`.
