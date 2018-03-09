import os
import time
from collections import deque
import pickle

from baselines.ddpg.ddpg import DDPG
import baselines.common.tf_util as U

from baselines import logger
import numpy as np
import tensorflow as tf
from mpi4py import MPI

from tradingstatemodel import TradingStateModel

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

def build_summaries(num_actions):
    episode_reward = tf.placeholder(dtype=tf.float32,
                                    shape=None)
    qfunc_loss = tf.placeholder(dtype=tf.float32,
                                shape=None)
    actions = tf.placeholder(dtype=tf.float32,
                             shape=[num_actions])
    prices = tf.placeholder(dtype=tf.float32,
                            shape=[num_actions])
    individual_reward = tf.placeholder(dtype=tf.float32,
                                       shape=None)
    individual_pnl = tf.placeholder(dtype=tf.float32,
                                    shape=None)
    individual_tc = tf.placeholder(dtype=tf.float32,
                                   shape=None)
    individual_estimated_q = tf.placeholder(dtype=tf.float32,
                                            shape=None)
    sum_episode_reward = tf.summary.scalar("Episode Reward", episode_reward)
    sum_qfunc_loss = tf.summary.scalar("Qfunc Loss", qfunc_loss)
    sum_actions = [tf.summary.scalar("Action-"+str(index), actions[index]) for
        index in range(num_actions)]
    sum_prices = [tf.summary.scalar("Price-"+str(index), prices[index]) for
        index in range(num_actions)]
    sum_individual_reward = tf.summary.scalar("Individual Reward", individual_reward)
    sum_individual_pnl = tf.summary.scalar("Individiual Pnl", individual_pnl)
    sum_individual_tc = tf.summary.scalar("Individual Tc", individual_tc)
    sum_individual_est_q = tf.summary.scalar("Estimated Q", individual_estimated_q)

    episode_summaries = tf.summary.merge([sum_episode_reward])
    individual_summaries = tf.summary.merge(sum_actions + sum_prices + [sum_individual_reward, 
                                                 sum_individual_pnl, sum_individual_tc, sum_individual_est_q])
    batch_summaries = tf.summary.merge([sum_qfunc_loss])

    return episode_summaries, individual_summaries, batch_summaries, \
        episode_reward, qfunc_loss, actions, prices, individual_reward, individual_pnl, individual_tc, individual_estimated_q

def train(env, nb_epochs, nb_epoch_cycles, render_eval, reward_scale, render, param_noise, actor, critic,
    normalize_returns, normalize_observations, critic_l2_reg, actor_lr, critic_lr, action_noise,
    popart, gamma, clip_norm, nb_train_steps, nb_rollout_steps, nb_eval_steps, batch_size, memory,
    tau=0.01, eval_env=None, param_noise_adaption_interval=50, tensorboard_directory=None, infer_directory=None):
    rank = MPI.COMM_WORLD.Get_rank()

    # assert (np.abs(env.action_space.low) == env.action_space.high).all()  # we assume symmetric actions.
    max_action = 1
    logger.info('scaling actions by {} before executing in env'.format(max_action))
    agent = DDPG(actor, critic, memory, env.observation_space.shape, env.action_space.shape,
        gamma=gamma, tau=tau, normalize_returns=normalize_returns, normalize_observations=normalize_observations,
        batch_size=batch_size, action_noise=action_noise, param_noise=param_noise, critic_l2_reg=critic_l2_reg,
        actor_lr=actor_lr, critic_lr=critic_lr, enable_popart=popart, clip_norm=clip_norm,
        reward_scale=reward_scale)
    logger.info('Using agent with the following configuration:')
    logger.info(str(agent.__dict__.items()))

    # Set up logging stuff only for a single worker.
    if rank == 0:
        saver = tf.train.Saver()
    else:
        saver = None

    if not os.path.exists(tensorboard_directory):
        os.makedirs(tensorboard_directory)
    else:
        for file in os.listdir(tensorboard_directory):
            file_path = os.path.join(tensorboard_directory, file)
            try:
                if os.path.isfile(file_path):
                    os.unlink(file_path)
            except Exception as e:
                print(e)
    if not os.path.exists(infer_directory):
        os.makedirs(infer_directory)
    else:
        for file in os.listdir(infer_directory):
            file_path = os.path.join(infer_directory, file)
            try:
                if os.path.isfile(file_path):
                    os.unlink(file_path)
            except Exception as e:
                print(e)
    step = 0
    episode = 0
    eval_episode_rewards_history = deque(maxlen=100)
    episode_rewards_history = deque(maxlen=100)
    with U.single_threaded_session() as sess:
        # Prepare everything.
        agent.initialize(sess)
        episode_summaries, individual_summaries, batch_summaries, \
            episode_reward_pl, qfunc_loss_pl, actions_pl, prices_pl, individual_reward_pl, \
            individual_pnl_pl, individual_tc_pl, individual_estimated_q_pl = build_summaries(env.action_space.shape[0])
        sess.graph.finalize()
        writer = tf.summary.FileWriter(tensorboard_directory, sess.graph)

        agent.reset()
        obs_state = env.reset()
        obs = obs_state.features
        if eval_env is not None:
            eval_obs = eval_env.reset().features
        done = False
        episode_reward = 0.
        episode_step = 0
        episodes = 0
        t = 0
        train_steps = 0

        epoch = 0
        start_time = time.time()

        epoch_episode_rewards = []
        epoch_episode_steps = []
        epoch_episode_eval_rewards = []
        epoch_episode_eval_steps = []
        epoch_start_time = time.time()
        epoch_actions = []
        epoch_qs = []
        epoch_episodes = 0
        for epoch in range(nb_epochs):
            for cycle in range(nb_epoch_cycles):
                # Perform rollouts.
                for t_rollout in range(nb_rollout_steps):
                    # Predict next action.
                    action, q = agent.pi(obs, apply_noise=True, compute_Q=True)
                    assert action.shape == env.action_space.shape

                    # Execute next action.
                    if rank == 0 and render:
                        env.render()
                    # assert max_action.shape == action.shape
                    new_obs_state, r, done, info = env.step(max_action * action)  # scale for execution in env (as far as DDPG is concerned, every action is in [-1, 1])
                    new_obs = new_obs_state.features
                    t += 1
                    if rank == 0 and render:
                        env.render()
                    episode_reward += r
                    episode_step += 1

                    summary = sess.run(individual_summaries, feed_dict={
                                        actions_pl: action,
                                        prices_pl: obs_state.price,
                                        individual_reward_pl: r,
                                        individual_pnl_pl: info['pnl'],
                                        individual_tc_pl: info['tc'],
                                        individual_estimated_q_pl: q[0, 0]
                                    })
                    writer.add_summary(summary, t)

                    # Book-keeping.
                    epoch_actions.append(action)
                    epoch_qs.append(q)
                    agent.store_transition(obs, action, r, new_obs, done)
                    obs = new_obs
                    obs_state = new_obs_state

                    if done:
                        # Episode done.
                        summary = sess.run(episode_summaries, feed_dict={
                                            episode_reward_pl: episode_reward
                                        })
                        writer.add_summary(summary, episodes)

                        epoch_episode_rewards.append(episode_reward)
                        episode_rewards_history.append(episode_reward)
                        epoch_episode_steps.append(episode_step)
                        episode_reward = 0.
                        episode_step = 0
                        epoch_episodes += 1
                        episodes += 1

                        agent.reset()
                        obs_state = env.reset()
                        obs = obs_state.features

                        if (episodes % 50) == 0:
                            print("HERE")
                            infer(env, agent, True, episodes, infer_directory)

                # Train.
                epoch_actor_losses = []
                epoch_critic_losses = []
                epoch_adaptive_distances = []
                for t_train in range(nb_train_steps):
                    train_steps += 1
                    # Adapt param noise, if necessary.
                    if memory.nb_entries >= batch_size and t_train % param_noise_adaption_interval == 0:
                        distance = agent.adapt_param_noise()
                        epoch_adaptive_distances.append(distance)

                    cl, al = agent.train()
                    summary = sess.run(batch_summaries, feed_dict={
                                        qfunc_loss_pl: cl
                                    })
                    writer.add_summary(summary, train_steps)
                    epoch_critic_losses.append(cl)
                    epoch_actor_losses.append(al)
                    agent.update_target_net()

                # Evaluate.
                eval_episode_rewards = []
                eval_qs = []
                if eval_env is not None:
                    eval_episode_reward = 0.
                    for t_rollout in range(nb_eval_steps):
                        eval_action, eval_q = agent.pi(eval_obs, apply_noise=False, compute_Q=True)
                        eval_obs, eval_r, eval_done, eval_info = eval_env.step(max_action * eval_action)  # scale for execution in env (as far as DDPG is concerned, every action is in [-1, 1])
                        eval_obs = eval_obs.features
                        if render_eval:
                            eval_env.render()
                        eval_episode_reward += eval_r

                        eval_qs.append(eval_q)
                        if eval_done:
                            eval_obs = eval_env.reset().features
                            eval_episode_rewards.append(eval_episode_reward)
                            eval_episode_rewards_history.append(eval_episode_reward)
                            eval_episode_reward = 0.

            mpi_size = MPI.COMM_WORLD.Get_size()
            # Log stats.
            # XXX shouldn't call np.mean on variable length lists
            duration = time.time() - start_time
            stats = agent.get_stats()
            combined_stats = stats.copy()
            print("EPOCH_EPISODE_REWARDS:", len(epoch_episode_rewards))
            combined_stats['rollout/return'] = np.mean(epoch_episode_rewards)
            combined_stats['rollout/return_history'] = np.mean(episode_rewards_history)
            combined_stats['rollout/episode_steps'] = np.mean(epoch_episode_steps)
            combined_stats['rollout/actions_mean'] = np.mean(epoch_actions)
            combined_stats['rollout/Q_mean'] = np.mean(epoch_qs)
            combined_stats['train/loss_actor'] = np.mean(epoch_actor_losses)
            combined_stats['train/loss_critic'] = np.mean(epoch_critic_losses)
            combined_stats['train/param_noise_distance'] = np.mean(epoch_adaptive_distances)
            combined_stats['total/duration'] = duration
            combined_stats['total/steps_per_second'] = float(t) / float(duration)
            combined_stats['total/episodes'] = episodes
            combined_stats['rollout/episodes'] = epoch_episodes
            combined_stats['rollout/actions_std'] = np.std(epoch_actions)
            # Evaluation statistics.
            if eval_env is not None:
                combined_stats['eval/return'] = eval_episode_rewards
                combined_stats['eval/return_history'] = np.mean(eval_episode_rewards_history)
                combined_stats['eval/Q'] = eval_qs
                combined_stats['eval/episodes'] = len(eval_episode_rewards)
            def as_scalar(x):
                if isinstance(x, np.ndarray):
                    assert x.size == 1
                    return x[0]
                elif np.isscalar(x):
                    return x
                else:
                    raise ValueError('expected scalar, got %s'%x)
            combined_stats_sums = MPI.COMM_WORLD.allreduce(np.array([as_scalar(x) for x in combined_stats.values()]))
            combined_stats = {k : v / mpi_size for (k,v) in zip(combined_stats.keys(), combined_stats_sums)}

            # Total statistics.
            combined_stats['total/epochs'] = epoch + 1
            combined_stats['total/steps'] = t

            for key in sorted(combined_stats.keys()):
                logger.record_tabular(key, combined_stats[key])
            logger.dump_tabular()
            logger.info('')
            logdir = logger.get_dir()
            if rank == 0 and logdir:
                if hasattr(env, 'get_state'):
                    with open(os.path.join(logdir, 'env_state.pkl'), 'wb') as f:
                        pickle.dump(env.get_state(), f)
                if eval_env and hasattr(eval_env, 'get_state'):
                    with open(os.path.join(logdir, 'eval_env_state.pkl'), 'wb') as f:
                        pickle.dump(eval_env.get_state(), f)

def infer(env, agent, train, episode, infer_directory):
    print("INFERRING episode:", episode)
    episode_length = env.datacontainer.train_length - 40
    tsm = TradingStateModel(datacontainer=env.datacontainer,
                            episode_length=episode_length,
                            is_training=True,
                            commission_percentage=env.commission_percentage)
    state = tsm.reset()
    prices = [state.price] # [episode_length]
    rewards = [0] # [episode_length]
    allocations = [state.portfolio_allocation] # [episode_length]

    for _ in range(episode_length):   
        action, _ = agent.pi(state.features, apply_noise=False, compute_Q=False)
        trans_state, reward, terminal, info = tsm.step(action)
        prices.append(trans_state.price)
        rewards.append(reward)
        allocations.append(trans_state.portfolio_allocation)
        state = trans_state

    prices = np.array(prices)
    rewards = np.array(rewards)
    allocations = np.array(allocations)

    f, axarr = plt.subplots(3, sharex=True)
    axarr[0].set_ylabel('Price')
    for ind in range(env.datacontainer.num_assets):
        axarr[0].plot(prices[:, ind])

    axarr[1].set_ylabel('Cumulative Reward')
    axarr[1].plot(np.cumsum(rewards))

    axarr[2].set_ylabel('Action')
    for ind in range(env.datacontainer.num_assets):
        axarr[2].plot(allocations[:, ind])

    dataset = 'Train' if train else 'Test'
    title = '{}, Total Reward: {}'.format(dataset,
                                          np.sum(rewards))
    plt.savefig(os.path.join(infer_directory, str(episode)+".png"))