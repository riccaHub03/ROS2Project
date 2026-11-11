"""Surgical robotics perception & grasping pipeline (KUKA LBR iiwa + ROS 2 + SAM 3 + Qwen).

Stages are independent modules, each runnable with ``python -m``:

    pipeline.plan                      LLM task planner
    pipeline.sim.capture               PyBullet scene + RGB-D render
    pipeline.sim.segment               SAM 3 on the simulated view (target + every instrument)
    pipeline.sim.overview              offscreen RGB-D + mask of every instrument on the full collection
    pipeline.sim.reference_pose        STL tray placement (certified pose)
    pipeline.sim.deproject             mask + depth -> 3D cloud and frames
    pipeline.real.capture / segment / deproject   real-image counterparts
    pipeline.grasp                     PCA zones + grasp pose

Importing this package is intentionally side-effect free (no heavy dependencies).
"""
