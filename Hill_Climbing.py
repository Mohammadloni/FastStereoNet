import os
import random
import sqlite3
import copy
import numpy as np
import tensorflow as tf
from lahc import LateAcceptanceHillClimber as climber
from Latency import Latency_estimation
import models.net_factory as nf
import datetime
import math
class HillClimber(climber):
    def __init__(self, data,state):
        self.num = 0
        self.data=data
        self.best = math.inf
        self.best_model=state
        os.mkdir(data.path+'\\HC')
        self.path = data.path + '\\HC\\bests.db'
        conn = sqlite3.connect(self.path)
        c = conn.cursor()
        c.execute('''CREATE TABLE bestss
                     (num int, arc text, acc real, t_flops real, Latency real, energy real)''')
        conn.commit()
        c = conn.cursor()
        c.execute('''CREATE TABLE _all_
                            (num int, arc text, acc real, t_flops real, Latency real, energy real)''')
        conn.commit()
        conn.close()
        super(HillClimber, self).__init__(state)

    def move(self):#Method of Mutating

        selection = random.choice([True, False])
        if selection:
            valids = []
            others = []
            layers = [['conv2d', 32, 'same', 3], ['conv2d', 32, 'same', 5], ['conv2d', 32, 'same', 7],
                      ['conv2d', 32, 'same', 11], ['batch', 0, 'none', 0], ['conc', 0, 'none', 0],
                      ['none', 0, 'none', 0]]
            x = self.state[0]
            for i in x:
                if i[0] == 'conv2d':
                    if i[2] == 'valid':
                        if i[3] > 3:
                            valids.append(x.index(i))
                        else:
                            pass
                    else:
                        others.append(x.index(i))
                else:
                    others.append(x.index(i))
            if len(valids) != 0:
                if random.random() < 0.2:
                    a = random.choice(valids)
                    b = random.randrange(3, x[a][3], 2)
                    temp = x[a][3]
                    x[a][3] = b
                    x.append(['conv2d', 32, 'valid', (temp - b) + 1])
                    print(self.state)


                elif random.random() >= 0.2 and random.random() < 0.7:
                    a = random.choice(others)
                    be = random.choice(layers)
                    x.remove(x[a])
                    x.append(be)
                else:
                    a = random.randint(0, len(x) - 1)
                    b = random.randint(0, len(x) - 1)
                    temp1 = x[a]
                    x[a] = x[b]
                    x[b] = temp1
            else:
                if random.random() < 0.5:
                    a = random.choice(others)
                    be = random.choice(layers)
                    x.remove(x[a])
                    x.append(be)
                else:
                    a = random.randint(0, len(x) - 1)
                    b = random.randint(0, len(x) - 1)
                    temp1 = x[a]
                    x[a] = x[b]
                    x[b] = temp1
            kernel_sum = 0
            num_node = 0
            for i in x:
                if i[0] == 'conv2d':
                    if i[2] == 'valid':
                        kernel_sum += i[3]
                        num_node += 1
            ex = kernel_sum - num_node
            if ex != 36:
                x.append(['conv2d', 32, 'valid', (36 - ex) + 1])
        else:
            initialized = False
            for i in self.state[1]:
                if i[0] != 'none':
                    initialized = True
            if initialized == False:
                self.state[1][len(self.state[1]) - 1] = ['conv2d', 32, 'valid', 37]
            else:
                valids = []
                others = []
                layers = [['conv2d', 32, 'same', 3], ['conv2d', 32, 'same', 5], ['conv2d', 32, 'same', 7],
                          ['conv2d', 32, 'same', 11], ['batch', 0, 'none', 0], ['conc', 0, 'none', 0],
                          ['none', 0, 'none', 0]]
                x = self.state[0]
                for i in x:
                    if i[0] == 'conv2d':
                        if i[2] == 'valid':
                            if i[3] > 3:
                                valids.append(x.index(i))
                            else:
                                pass
                        else:
                            others.append(x.index(i))
                    else:
                        others.append(x.index(i))
                if len(valids) != 0:
                    if random.random() < 0.2:
                        a = random.choice(valids)
                        b = random.randrange(3, x[a][3], 2)
                        temp = x[a][3]
                        x[a][3] = b
                        x.append(['conv2d', 32, 'valid', (temp - b) + 1])
                    elif random.random() >= 0.2 and random.random() < 0.7:
                        a = random.choice(others)
                        be = random.choice(layers)
                        x.remove(x[a])
                        x.append(be)
                    else:
                        a = random.randint(0, len(x) - 1)
                        b = random.randint(0, len(x) - 1)
                        temp1 = x[a]
                        x[a] = x[b]
                        x[b] = temp1
                else:
                    if random.random() < 0.5:
                        a = random.choice(others)
                        be = random.choice(layers)
                        x.remove(x[a])
                        x.append(be)
                    else:
                        a = random.randint(0, len(x) - 1)
                        b = random.randint(0, len(x) - 1)
                        temp1 = x[a]
                        x[a] = x[b]
                        x[b] = temp1
                kernel_sum = 0
                num_node = 0
                for i in x:
                    if i[0] == 'conv2d':
                        if i[2] == 'valid':
                            kernel_sum += i[3]
                            num_node += 1
                ex = kernel_sum - num_node
                if ex != 36:
                    x.append(['conv2d', 32, 'valid', (36 - ex) + 1])
        return self.energy()

    def energy(self):#Method of Computing each model's Energy
        has = False
        for i in self.state[1]:
            if i[0] != 'none':
                has = True

        if has == True:
            for x in self.state:
                kernel_sum = 0
                num_node = 0
                for i in x:
                    if i[0] == 'conv2d':
                        if i[2] == 'valid':
                            kernel_sum += i[3]
                            num_node += 1
                ex = kernel_sum - num_node
                if ex != 36:
                    raise ValueError('this is not appropriante')
        else:
            kernel_sum = 0
            num_node = 0
            for i in self.state[0]:
                if i[0] == 'conv2d':
                    if i[2] == 'valid':
                        kernel_sum += i[3]
                        num_node += 1
            ex = kernel_sum - num_node
            if ex != 36:
                raise ValueError('this is not appropriante')
        t_flops =train(self.data,self.state, self.num)
        acc = evaluate(self.data,self.state, self.num)
        Latency=Latency_estimation(self.state)
        if acc == 0.0:
            e = math.inf
        else:
            #e = t_flops / acc
            e=Latency/acc
        statea = str(self.state)
        conn = sqlite3.connect(self.path)
        c = conn.cursor()
        c.execute('''INSERT INTO _all_ VALUES (?,?,?,?,?,?)''', [self.num, statea, acc, t_flops,Latency, e])
        conn.commit()
        conn.close()
        wr='HC'+str(self.num)+'=\t' + str(datetime.datetime.now()) + '\n'
        with open(self.data.timefile, "a+") as handle:
            print(wr, file=handle)

        if e < self.best:
            self.best_model=copy.deepcopy(self.state)
            conn = sqlite3.connect(self.path)
            c = conn.cursor()
            c.execute('''INSERT INTO bestss VALUES (?,?,?,?,?,?)''', [self.num, statea, acc, t_flops,Latency, e])
            conn.commit()
            conn.close()
            self.best = e
        self.num = self.num + 1
        print(self.state)
        return e


def train(data,state, number):#Partialy Computing created models
    patch_size=37
    num_channels=3
    disp_range=201
    net_type='win37_dep9'
    path = data.path + '/HC/' + str(number)
    if not os.path.exists(path):
        os.makedirs(path)
    tf.reset_default_graph()
    run_meta = tf.RunMetadata()
    g = tf.Graph()
    strategy = tf.distribute.MirroredStrategy()
    with strategy.scope():
        with g.as_default():

            limage = tf.placeholder(tf.float32, [None, patch_size, patch_size, num_channels], name='limage')
            rimage = tf.placeholder(tf.float32,
                                    [None, patch_size,patch_size + disp_range - 1, num_channels],
                                    name='rimage')
            targets = tf.placeholder(tf.float32, [None, disp_range], name='targets')

            snet = nf.create(limage, rimage, targets, state,net_type)

            loss = snet['loss']
            train_step = snet['train_step']
            session = tf.InteractiveSession()
            session.run(tf.global_variables_initializer())
            saver = tf.train.Saver(max_to_keep=1)

            acc_loss = tf.placeholder(tf.float32, shape=())
            loss_summary = tf.summary.scalar('loss', acc_loss)
            train_writer = tf.summary.FileWriter(path + '/training', g)

            saver = tf.train.Saver(max_to_keep=1)
            losses = []
            summary_index = 1
            lrate = 1e-2

            for it in range(1, data.num_iter):
                lpatch, rpatch, patch_targets = data.dhandler.next_batch()

                train_dict = {limage: lpatch, rimage: rpatch, targets: patch_targets,
                              snet['is_training']: True, snet['lrate']: lrate}
                _, mini_loss = session.run([train_step, loss], feed_dict=train_dict)
                losses.append(mini_loss)

                if it % 10 == 0:
                    print('Loss at step: %d: %.6f' % (it, mini_loss))
                    saver.save(session, os.path.join(path, 'model.ckpt'), global_step=snet['global_step'])
                    train_summary = session.run(loss_summary,
                                                feed_dict={acc_loss: np.mean(losses)})
                    train_writer.add_summary(train_summary, summary_index)
                    summary_index += 1
                    train_writer.flush()
                    losses = []

                if it == 24000:
                    lrate = lrate / 5.
                elif it > 24000 and (it - 24000) % 8000 == 0:
                    lrate = lrate / 5.
        opts = tf.profiler.ProfileOptionBuilder.float_operation()
        flops = tf.profiler.profile(g, run_meta=run_meta, cmd='op', options=opts)
        if flops is not None:
            t_flops = flops.total_float_ops
    return t_flops


def evaluate(data,state, number):#Evaluate the model's accuracy
    patch_size = 37
    num_channels = 3
    disp_range = 201
    eval_size=200
    net_type = 'win37_dep9'
    lpatch, rpatch, patch_targets = data.dhandler.evaluate()
    labels = np.argmax(patch_targets, axis=1)
    path = data.path + '/HC/' + str(number)
    with tf.Session() as session:
        limage = tf.placeholder(tf.float32, [None, patch_size,patch_size, num_channels], name='limage')
        rimage = tf.placeholder(tf.float32,
                                [None, patch_size, patch_size +disp_range - 1, num_channels],
                                name='rimage')
        targets = tf.placeholder(tf.float32, [None, disp_range], name='targets')

        snet = nf.create(limage, rimage, targets, state, net_type)
        prod = snet['inner_product']
        predicted = tf.argmax(prod, axis=1)
        acc_count = 0

        saver = tf.train.Saver()
        saver.restore(session, tf.train.latest_checkpoint(path))

        for i in range(0, lpatch.shape[0], eval_size):
            eval_dict = {limage: lpatch[i: i + eval_size],
                         rimage: rpatch[i: i + eval_size], snet['is_training']: False}
            pred = session.run([predicted], feed_dict=eval_dict)
            acc_count += np.sum(np.abs(pred - labels[i: i +eval_size]) <= 3)
            print('iter. %d finished, with %d correct (3-pixel error)' % (i + 1, acc_count))

            print('accuracy: %.3f' % ((acc_count / lpatch.shape[0]) * 100))
    tf.reset_default_graph()
    return ((acc_count / lpatch.shape[0]) * 100)
def Hill_Climber(data,idle_fraction,init):
    #Set LAHC HyperParameters
    climber.history_length = 50
    climber.updates_every = 10
    climber.steps_minimum=100
    climber.steps_idle_fraction=idle_fraction
    prob_slice = HillClimber(data,init)
    prob_slice.copy_strategy = 'slice'
    prob_slice.run()
    return prob_slice.best_model#Return Best model