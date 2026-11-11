"""CPU visualization of nominal Robotiq control lines, not collision geometry."""
import numpy as np


def gripper_lines(pose):
    # Same width/depth convention as config/grippers/robotiq_2f_140.yaml.
    width, depth = 0.13603458, 0.195
    local = np.array([[width/2, 0, depth/2], [width/2, 0, depth], [0, 0, depth],
                      [0, 0, 0], [0, 0, depth], [-width/2, 0, depth], [-width/2, 0, depth/2]])
    t = np.asarray(pose)
    return local @ t[:3, :3].T + t[:3, 3]


def plot_candidates(path, points, candidates, limit=20):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig = plt.figure(figsize=(10, 7))
    ax = fig.add_subplot(111, projection="3d")
    stride = max(1, len(points) // 4000)
    ax.scatter(*points[::stride].T, s=1, c="gray", alpha=0.5)
    for i, candidate in enumerate(candidates[:limit]):
        line = gripper_lines(candidate["T_base_gripper"])
        ax.plot(*line.T, color="limegreen" if i == 0 else "steelblue", alpha=1 if i == 0 else 0.25)
    tcp = np.asarray(candidates[0]["T_base_tcp"])
    ax.scatter(*tcp[:3, 3], c="red", s=40, label="selected nominal TCP")
    ax.set(xlabel="base X [m]", ylabel="base Y [m]", zlabel="base Z [m]",
           title=f"GraspGen / Robotiq 2F-140: {len(candidates)} candidates; physical grasp UNVALIDATED")
    limits = np.array([ax.get_xlim(), ax.get_ylim(), ax.get_zlim()])
    center = limits.mean(axis=1)
    radius = np.ptp(limits, axis=1).max() / 2
    ax.set_xlim(center[0]-radius, center[0]+radius)
    ax.set_ylim(center[1]-radius, center[1]+radius)
    ax.set_zlim(center[2]-radius, center[2]+radius)
    ax.set_box_aspect([1, 1, 1])
    ax.legend()
    fig.tight_layout()
    fig.savefig(path, dpi=140)
    plt.close(fig)
