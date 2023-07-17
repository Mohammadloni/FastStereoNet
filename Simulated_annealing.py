import os
import random
import sqlite3
import time

import numpy as np
import tensorflow as tf
from simanneal import Annealer
from termcolor import colored
from Latency import Latency_estimation
import models.net_factory as nf
import datetime
import math
class SimulatedAnnealer(Annealer):
    def __init__(self, data,state):
        self.data=data
        self.num = 0
        self.last_model=None
        self.last_t_flops=None
        self.last_acc=None
        self.last_latency=None
        self.first=True
        self.weights={}
        self.best=math.inf
        os.mkdir(data.path+'\\SA')
        self.path=data.path+'\\SA\\bests.db'
        conn=sqlite3.connect(self.path)
        c=conn.cursor()
        c.execute('''CREATE TABLE bestss
                     (num int, arc text, acc real, t_flops real,Latency real, energy real)''')
        conn.commit()
        c = conn.cursor()
        c.execute('''CREATE TABLE _all_
                            (num int, arc text, acc real, t_flops real,Latency real, energy real)''')
        conn.commit()
        conn.close()
        super(SimulatedAnnealer, self).__init__(state)

    def move(self):#Method of Mutating
        selection=random.choice([True,False])
        if selection:
            valids = []
            others = []
            layers = [['conv2d', 32, 'same', 3], ['conv2d', 32, 'same', 5], ['conv2d', 32, 'same', 7],
                      ['conv2d', 32, 'same', 11], ['batch', 0, 'none', 0],['conc',0,'none',0],
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
        else:

            initialized=False
            for i in self.state[1]:
                if i[0]!='none':
                    initialized=True
            if initialized==False:
                self.state[1][len(self.state[1])-1]=['conv2d',32,'valid',37]
            else:
                valids = []
                others = []
                layers = [['conv2d', 32, 'same', 3], ['conv2d', 32, 'same', 5], ['conv2d', 32, 'same', 7],
                          ['conv2d', 32, 'same', 11], ['batch', 0, 'none', 0],['conc',0,'none',0],
                          ['none', 0, 'none', 0]]
                x = self.state[1]
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


    def energy(self):#method of computing created model's energy
        if self.state==self.last_model:
            acc=self.last_acc
            t_flops=self.last_t_flops
            latency=self.last_latency
        else:
            has=False
            for i in self.state[1]:
                if i[0]!='none':
                    has=True


            if has==True:
                for x in self.state:
                    kernel_sum = 0
                    num_node = 0
                    for i in x:
                        if i[0]=='conv2d':
                            if i[2]=='valid':
                                kernel_sum+=i[3]
                                num_node+=1
                    ex=kernel_sum-num_node
                    if ex!=36:
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
            t_flops,weights,time=train(self.data,self.state, self.num,self.weights,self.first)
            self.first=False
            self.weights=weights
            acc = evaluate(self.data,self.state,self.num)
            latency=Latency_estimation(self.state)
            print('latency',latency)
        if acc==0.0:
            e=math.inf
        else:
            #e=t_flops/acc
            e=latency/acc
        statea=str(self.state)
        conn = sqlite3.connect(self.path)
        c = conn.cursor()
        c.execute('''INSERT INTO _all_ VALUES (?,?,?,?,?,?)''', [self.num, statea, acc, t_flops,latency, e])
        conn.commit()
        conn.close()
        with open(self.data.timefile, "a+") as handle:
            print('SA'+str(self.num)+'=\t' + str(datetime.datetime.now()) + '\n', file=handle)

        if e<self.best:
            conn = sqlite3.connect(self.path)
            c = conn.cursor()
            c.execute('''INSERT INTO bestss VALUES (?,?,?,?,?,?)''',[self.num,statea,acc,t_flops,latency,e])
            conn.commit()
            conn.close()
            self.best=e
        self.num = self.num + 1
        import copy
        self.last_model=copy.deepcopy(self.state)

        self.last_t_flops=copy.deepcopy(t_flops)
        self.last_acc=copy.deepcopy(acc)
        self.last_latency=copy.deepcopy(latency)

        return e

def train(data,state, number, weights,first):#Partialy Training the model
    count1 = 0
    count2 = 0
    patch_size = 37
    num_channels = 3
    disp_range = 201
    net_type = 'win37_dep9'
    path = data.path + '/SA/' + str(number)
    if not os.path.exists(path):
        os.makedirs(path)
    tf.reset_default_graph()
    run_meta = tf.RunMetadata()
    g = tf.Graph()
    strategy=tf.distribute.MirroredStrategy()
    with strategy.scope():
        with g.as_default():

            limage = tf.placeholder(tf.float32, [None, patch_size, patch_size, num_channels], name='limage')
            rimage = tf.placeholder(tf.float32,
                                    [None, patch_size,patch_size + disp_range - 1, num_channels],
                                    name='rimage')
            targets = tf.placeholder(tf.float32, [None,disp_range], name='targets')

            snet = nf.create(limage, rimage, targets, state, net_type)

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
            Total_time=0
            if first == False:
                check=False
                y=g._collections['trainable_variables']
                for ele1 in y:
                    if ele1.name in weights.keys():
                        try:
                            op=tf.assign(ele1._variable,weights[ele1.name])
                            session.run(op)
                            count1+=1
                        except:
                            count2+=1
                            pass

            for it in range(1, data.num_iter):
                lpatch, rpatch, patch_targets = data.dhandler.next_batch()

                train_dict = {limage: lpatch, rimage: rpatch, targets: patch_targets,
                              snet['is_training']: True, snet['lrate']: lrate}
                t1 = int(round(time.time() * 1000))
                _, mini_loss = session.run([train_step, loss], feed_dict=train_dict)
                t2 = int(round(time.time() * 1000))
                losses.append(mini_loss)
                Total_time+=t2-t1

                if it % 10 == 0:
                    print('Loss at step: %d: %.6f' % (it, mini_loss)) #please us me later
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
                weights={}
                x=g._collections['trainable_variables']
                for ele in x:
                    weights.update({ele.name:ele._variable.eval()})
        opts = tf.profiler.ProfileOptionBuilder.float_operation()
        flops = tf.profiler.profile(g, run_meta=run_meta, cmd='op', options=opts)
        if flops is not None:
            t_flops=flops.total_float_ops
    file1=open(path+'\\g.txt','a+')
    file1.write(str(g._nodes_by_name))
    file1.close()
    return t_flops,weights,time


def evaluate(data,state,number):#Computing the Accuracy
    patch_size = 37
    num_channels = 3
    disp_range = 201
    net_type = 'win37_dep9'
    eval_size = 200
    lpatch, rpatch, patch_targets = data.dhandler.evaluate()
    labels = np.argmax(patch_targets, axis=1)
    path = data.path+ '/SA/' + str(number)
    with tf.Session() as session:
        limage = tf.placeholder(tf.float32, [None, patch_size, patch_size, num_channels], name='limage')
        rimage = tf.placeholder(tf.float32,
                                [None, patch_size, patch_size + disp_range - 1, num_channels],
                                name='rimage')
        targets = tf.placeholder(tf.float32, [None, disp_range], name='targets')

        snet = nf.create(limage, rimage, targets, state,net_type)
        prod = snet['inner_product']
        predicted = tf.argmax(prod, axis=1)
        acc_count = 0

        saver = tf.train.Saver()
        saver.restore(session, tf.train.latest_checkpoint(path))

        for i in range(0, lpatch.shape[0], eval_size):
            eval_dict = {limage: lpatch[i: i + eval_size],
                            rimage: rpatch[i: i + eval_size], snet['is_training']: False}
            pred = session.run([predicted], feed_dict=eval_dict)
            acc_count += np.sum(np.abs(pred - labels[i: i + eval_size]) <= 3)
            print('iter. %d finished, with %d correct (3-pixel error)' % (i + 1, acc_count))

            print('accuracy: %.3f' % ((acc_count / lpatch.shape[0]) * 100))
    tf.reset_default_graph()
    return ((acc_count / lpatch.shape[0]) * 100)
def Sim_Annealer(data,init):#initialize Simulated annealing 
    tsp = SimulatedAnnealer(data,init)
    tsp.set_schedule(tsp.auto(0.01, 10))
    tsp.copy_strategy = "slice"
    state, e = tsp.anneal()
    return state