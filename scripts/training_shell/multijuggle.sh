total_frames=1_000_000_000
algorithm="mappo"  # mappo, maddpg, happo, mat, qmix
seed=0  # 0, 1, 2

CUDA_VISIBLE_DEVICES=0 python ../train.py headless=true \
    total_frames=${total_frames} \
    task=MultiJuggleVolleyball \
    task.action_transform=PIDrate_FM \
    task.drone_model=Air \
    task.env.num_envs=4096 \
    task.ball_mass=0.005 \
    task.ball_radius=0.1 \
    task.reward_action_smoothness_weight=0 \
    eval_interval=500 \
    save_interval=500 \
    algo=${algorithm} \
    seed=${seed} \
    algo.critic_input=state \
    wandb.mode=online \