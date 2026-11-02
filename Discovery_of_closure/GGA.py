import random
import heapq
import numpy as np
import pickle
from tqdm import tqdm
class GGA():
    def __init__(self,x,t,u,u_x,u_xx,u_xxx,u_t,u_tt,epi,dim=1,delete_num=10,creterion='standard',temp_tag=""):
        self.dim=dim
        self.max_length = 5
        self.partial_prob = 0.6
        self.genes_prob = 0.6
        self.mutate_rate = 0.3
        self.delete_rate = 0.5
        self.add_rate = 0.4
        self.pop_size = 400
        self.n_generations = 200
        self.delete_num=delete_num
        self.u=u
        self.u_x=u_x
        self.u_xx=u_xx
        self.u_xxx=u_xxx
        self.u_t=u_t
        self.u_tt=u_tt
        self.x=x
        self.t=t
        self.dx=x[1]-x[0]
        self.dt=t[1]-t[0]
        self.nx=x.shape[0]
        self.nt=t.shape[0]
        self.total_delete=(x.shape[0]-delete_num)*(t.shape[0]-delete_num)
        self.epi=epi
        self.creterion=creterion
        suffix = "_{}".format(temp_tag) if temp_tag else ""
        self.temp_ga_path = "temp_save/best_save_GA{}.pkl".format(suffix)
        self.temp_diff_path = "temp_save/best_save_diff_GA{}.pkl".format(suffix)

    def Delete_boundary(self,u,nx,nt):
        un=u.reshape(nx,nt)
        un_del=un[5:nx-5,5:nt-5]
        return un_del.reshape((nx-10)*(nt-10),1)

    def FiniteDiff_x(self,un,d):
        #u=[nx,nt]
        #用二阶微分计算d阶微分，不过在三阶以上准确性会比较低
        #u是需要被微分的数据
        #dx是网格的空间大小
        u=un.T
        dx=self.dx
        nt,nx=u.shape
        ux=np.zeros([nt,nx])

        if d==1:

            ux[:,1:nx-1]=(u[:,2:nx]-u[:,0:nx-2])/(2*dx)
            ux[:,0]=(-3.0/2*u[:,0]+2*u[:,1]-u[:,2]/2)/dx
            ux[:,nx-1]=(2.0/2*u[:,nx-1]-2*u[:,nx-2]+u[:,nx-3]/2)/dx
            return  ux.T

        if d==2:
            ux[:,1:nx-1]=(u[:,2:nx]-2*u[:,1:nx-1]+u[:,0:nx-2])/dx**2
            ux[:,0]=(2*u[:,0]-5*u[:,1]+4*u[:,2]-u[:,3])/dx**2
            ux[:,nx-1]=(2*u[:,nx-1]-5*u[:,nx-2]+4*u[:,nx-3]-u[:,nx-4])/dx**2
            return ux.T

        if d==3:
            ux[:,2:nx-2]=(u[:,4:nx]/2-u[:,3:nx-1]+u[:,1:nx-3]-u[:,0:nx-4]/2)/dx**3
            ux[:,0]=(-2.5*u[:,0]+9*u[:,1]-12*u[:,2]+7*u[:,3]-1.5*u[:,4])/dx**3
            ux[:,1]=(-2.5*u[:,1]+9*u[:,2]-12*u[:,3]+7*u[:,4]-1.5*u[:,5])/dx**3
            ux[:,nx-1]=(2.5*u[:,nx-1]-9*u[:,nx-2]+12*u[:,nx-3]-7*u[:,nx-4]+1.5*u[:,nx-5])/dx**3
            ux[:,nx-2]=(2.5*u[:,nx-2]-9*u[:,nx-3]+12*u[:,nx-4]-7*u[:,nx-5]+1.5*u[:,nx-6])/dx**3
            return ux.T

        if d>3:
            return GGA.FiniteDiff_x(GGA.FiniteDiff_x(u,dx,3),dx,d-3)

    def random_diff_module(self):
        if self.dim==1:
            diff_y=0
        if self.dim==2:
            diff_y=random.randint(0,3)
        diff_x=random.randint(0,3)
        genes_module = [diff_y,diff_x]
        return genes_module

    def random_module(self):
        genes_module=[]
        genes_diff_module = GGA.random_diff_module(self)
        if self.creterion=='standard':
            for i in range(self.max_length):
                a=random.randint(0,2)
                genes_module.append(a)
                prob=random.uniform(0,1)
                if prob>self.partial_prob:
                    break
        if self.creterion=='Galieo':
            a = random.randint(1, 2)
            genes_module.append(a)
        return genes_module,genes_diff_module

    def random_genome(self):
        genes=[]
        gene_diff=[]

        if self.creterion=='Galieo':
            genes.append([0,0])
            gene_diff.append([0,1])

        for i in range(self.max_length):
            gene_random,gene_random_diff=GGA.random_module(self)
            genes.append(sorted(gene_random))
            gene_diff.append((gene_random_diff))
            prob=random.uniform(0,1)
            if prob>self.genes_prob:
                break
        return genes,gene_diff

    def translate_DNA(self,gene,gene_left):
        gene_translate=np.ones([self.total_delete,1])
        length_penalty_coef=0
        for k in range(len(gene)):
            gene_module=gene[k]
            gene_left_module=gene_left[k]
            length_penalty_coef+=len(gene_module)
            module_out=np.ones([self.u.shape[0],self.u.shape[1]])
            for i in gene_module:
                if i==0:
                    temp=self.u
                if i==1:
                    temp=self.u_x
                if i==2:
                    temp=self.u_xx
                if i==3:
                    temp=self.u_xxx
                module_out*=temp
            un=module_out.reshape(self.nx,self.nt)
            if gene_left_module[1]>0:
                un_x=GGA.FiniteDiff_x(self,un,d=gene_left_module[1])
                un=un_x
            un = GGA.Delete_boundary(self,un, self.nx, self.nt)
            module_out=un.reshape([self.total_delete,1])
            gene_translate=np.hstack((gene_translate,module_out))
        gene_translate=np.delete(gene_translate,[0],axis=1)
        return gene_translate,length_penalty_coef

    def get_fitness(self,gene_translate,length_penalty_coef):
        u_t=self.u_t
        u_t_new=u_t.reshape([self.nx,self.nt])
        u_t=GGA.Delete_boundary(self,u_t_new,self.nx,self.nt).reshape(self.total_delete,1)
        u, d, v = np.linalg.svd(np.hstack((u_t, gene_translate)), full_matrices=False)
        coef_NN = v.T[:, -1] / (v.T[:, -1][0] + 1e-8)
        coef=-coef_NN[1:].reshape(coef_NN.shape[0]-1,1)
        res = u_t-np.dot(gene_translate,coef)

        u_tt=self.u_tt
        u_tt_new=u_tt.reshape([self.nx,self.nt])
        u_tt=GGA.Delete_boundary(self,u_tt_new,self.nx,self.nt).reshape(self.total_delete,1)
        u, d, v = np.linalg.svd(np.hstack((u_tt, gene_translate)), full_matrices=False)
        coef_NN = v.T[:, -1] / (v.T[:, -1][0] + 1e-8)
        coef_tt=-coef_NN[1:].reshape(coef_NN.shape[0]-1,1)
        res_tt = u_tt-np.dot(gene_translate,coef_tt)
        MSE_true = np.sum(np.array(res) ** 2) / self.total_delete
        MSE_true_tt = np.sum(np.array(res_tt) ** 2) / self.total_delete

        if MSE_true<MSE_true_tt:
            name='u_t'
            MSE=MSE_true+self.epi*length_penalty_coef
            coef=coef
            return coef,MSE,MSE_true,name


        if MSE_true>MSE_true_tt:
            name='u_tt'
            MSE = MSE_true_tt + self.epi * length_penalty_coef
            return coef_tt, MSE, MSE_true_tt, name



    def cross_over(self):
        Chrom,Chrom_diff, size_pop = self.Chrom,self.Chrom_diff, self.n_generations
        Chrom1, Chrom2 = Chrom[::2], Chrom[1::2]
        Chrom1_diff, Chrom2_diff = Chrom_diff[::2], Chrom_diff[1::2]
        for i in range(int(size_pop / 2)):
            if self.creterion=='standard':
                n1= np.random.randint(0, len(Chrom1[i]))
                n2=np.random.randint(0, len(Chrom2[i]))

            if self.creterion=='Galieo':
                if len(Chrom1[i])>1 and len(Chrom2[i])>1:
                    n1 = np.random.randint(1, len(Chrom1[i]))
                    n2 = np.random.randint(1, len(Chrom2[i]))
                else:
                    continue


            father=Chrom1[i][n1].copy()
            mother=Chrom2[i][n2].copy()

            father_diff=Chrom1_diff[i][n1].copy()
            mother_diff=Chrom2_diff[i][n2].copy()

            Chrom1[i][n1]=mother
            Chrom2[i][n2]=father

            Chrom1_diff[i][n1] = mother_diff
            Chrom2_diff[i][n2] = father_diff

        Chrom[::2], Chrom[1::2] = Chrom1, Chrom2
        Chrom_diff[::2], Chrom_diff[1::2] = Chrom1_diff, Chrom2_diff
        self.Chrom = Chrom
        self.Chrom_diff = Chrom_diff
        return self.Chrom,self.Chrom_diff

    def mutation(self):
        Chrom,Chrom_diff, size_pop = self.Chrom,self.Chrom_diff, self.pop_size

        for i in range(size_pop):
            n1 = np.random.randint(0, len(Chrom[i]))

            # ------------add module---------------
            prob = np.random.uniform(0, 1)
            if prob < self.add_rate:
                add_Chrom,add_Chrom_diff = GGA.random_module(self)
                if add_Chrom not in Chrom[i]:
                    Chrom[i].append(add_Chrom)
                    Chrom_diff[i].append(add_Chrom_diff)

            # --------delete module----------------
            prob = np.random.uniform(0, 1)
            if prob < self.mutate_rate:
                if len(Chrom[i]) > 1:
                    if self.creterion=='standard':
                        delete_index = np.random.randint(0, len(Chrom[i]))
                    if self.creterion=='Galieo':
                        delete_index = np.random.randint(1, len(Chrom[i]))
                    Chrom[i].pop(delete_index)
                    Chrom_diff[i].pop(delete_index)

            # ------------gene mutation------------------
            if self.creterion=='standard':
                prob = np.random.uniform(0, 1)
                if prob < self.mutate_rate:
                    if len(Chrom[i]) > 0:
                        n1 = np.random.randint(0, len(Chrom[i]))
                        if len(Chrom[i])>1:
                            n1 = np.random.randint(1, len(Chrom[i]))
                        else:
                            continue
                        n2 = np.random.randint(0, len(Chrom[i][n1]))
                        Chrom[i][n1][n2] = random.randint(0,3)
                        Chrom[i][n1]=GGA.random_diff_module(self)


        self.Chrom = Chrom
        self.Chrom_diff=Chrom_diff
        return self.Chrom,self.Chrom_diff


    def select(self):  # nature selection wrt pop's fitness
        Chrom, Chrom_diff,size_pop = self.Chrom,self.Chrom_diff, self.pop_size
        new_Chrom=[]
        new_fitness=[]
        new_Chrom_diff=[]
        new_coef=[]
        new_name=[]

        fitness_list = []
        coef_list=[]
        name_list=[]

        for i in range(size_pop):
            gene_translate, length_penalty_coef = GGA.translate_DNA(self, Chrom[i],Chrom_diff[i])
            coef, MSE, MSE_true,name = GGA.get_fitness(self, gene_translate, length_penalty_coef)
            fitness_list.append(MSE)
            coef_list.append(coef)
            name_list.append(name)
        re1 = list(map(fitness_list.index, heapq.nsmallest(int(size_pop/2), fitness_list)))

        for index in re1:
            new_Chrom.append(Chrom[index])
            new_Chrom_diff.append(Chrom_diff[index])
            new_fitness.append(fitness_list[index])
            new_coef.append(coef_list[index])
            new_name.append(name_list[index])
        for index in range(int(size_pop/2)):
            new,new_diff=GGA.random_genome(self)
            new_Chrom.append(new)
            new_Chrom_diff.append(new_diff)


        self.Chrom=new_Chrom
        self.Chrom_diff=new_Chrom_diff
        self.Fitness=new_fitness
        self.coef=new_coef
        self.name=new_name
        return self.Chrom,self.Fitness,self.coef,self.name

    def delete_duplicates(self):
        Chrom,Chrom_diff, size_pop = self.Chrom,self.Chrom_diff, self.pop_size
        for i in range(size_pop):
            new_genome=[]
            new_genome_diff=[]
            for j in range(len(Chrom[i])):
                if sorted(Chrom[i][j]) not in new_genome:
                    new_genome.append(sorted(Chrom[i][j]))
                    new_genome_diff.append(Chrom_diff[i][j])
            Chrom[i]=new_genome
            Chrom_diff[i]=new_genome_diff
        self.Chrom=Chrom
        self.Chrom_diff=Chrom_diff
        return self.Chrom,self.Chrom_diff


    def evolution(self):
        self.Chrom = []
        self.Chrom_diff=[]
        self.Fitness=[]
        for iter in range(self.pop_size):
            intial_genome,intial_genome_diff =GGA.random_genome(self)
            self.Chrom.append(intial_genome)
            self.Chrom_diff.append(intial_genome_diff)
            gene_translate, length_penalty_coef=GGA.translate_DNA(self,intial_genome,intial_genome_diff)
            coef, MSE,MSE_true,name=GGA.get_fitness(self,gene_translate,length_penalty_coef)
            self.Fitness.append(MSE)
        GGA.delete_duplicates(self)

        for iter in tqdm(range(self.n_generations)):
            #print(f'--------{iter}----------------')
            pickle.dump(self.Chrom.copy()[0],open(self.temp_ga_path, 'wb'))
            pickle.dump(self.Chrom_diff.copy()[0], open(self.temp_diff_path, 'wb'))

            best =self.Chrom.copy()[0]
            best_nc=self.Chrom_diff.copy()[0]
            GGA.cross_over(self)
            GGA.mutation(self)
            GGA.delete_duplicates(self)
            best=pickle.load( open(self.temp_ga_path, 'rb'))
            best_diff=pickle.load(open(self.temp_diff_path, 'rb'))

            self.Chrom[0]=best
            self.Chrom_diff[0]=best_diff
            GGA.select(self)

            # print(f'The best Chrom: {self.Chrom[0]}')
            # print(f'The best diff:  {self.Chrom_diff[0]}')
            # print(f'The best coef:  \n{self.coef[0]}')
            # print(f'The best fitness: {self.Fitness[0]}')
            # print(f'The best name: {self.name[0]}\r')

        return self.Chrom[0],self.Chrom_diff,self.coef[0],self.Fitness[0]
