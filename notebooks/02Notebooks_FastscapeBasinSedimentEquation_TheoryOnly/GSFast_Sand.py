import numpy as np
import xsimlab as xs
import GSFast_Functions
#The Grain size Class Main that will call these Functions
#Calculate a grain size fining grid NEW METHOD (Jan 31, 2022)
from fastscape.processes import (TotalErosion, MultipleFlowRouter, RasterGrid2D, Bedrock) #These are the Fastscape Processes needed to calculate the grain size. 

@xs.process   
class SandGSN:
    #Inputs from Fastscape Processes
    E = xs.foreign(TotalErosion, 'rate',intent="in") #description="The Erosion Rate from Fastscape Total Erosion process", attrs={"units": "m/yr"})
    stack=xs.foreign(MultipleFlowRouter, 'stack',intent="in") #description="The flow process stack in Fastscape", attrs={"units": "Dimensionless"})
    rec=xs.foreign(MultipleFlowRouter, 'receivers',intent="in") #description="The list of recievers from the flow process in Fastscape", attrs={"units": "Dimensionless"})
    nrec=xs.foreign(MultipleFlowRouter, 'nb_receivers',intent="in") #Specific to multiflow- list of multiple recievers
    rec_weights=xs.foreign(MultipleFlowRouter, 'weights',intent="in") #Specific to multiflow - list of weights for each reciever
    gridshape=xs.foreign(RasterGrid2D, 'shape',intent="in")# description="The Fastscape cell grid mesh shape", attrs={"units": "Dimensionless"})
    BedDepth = xs.foreign(Bedrock, 'depth',intent="in") #description="The bedrock elevation from the bedrock process in Fastscape", attrs={"units": "m"})
    #User defined inputs (Eg: Fedele and Paola (2007) Coefficients)
    D0=xs.variable(dims=(),intent="in",description='The bedrock source D50 grain size', attrs={"units": "mm"})
    C3=xs.variable(dims=(),intent="in",description='C3 (grain size calculation constant) described in Fedele and Paola (2007) and Duller et al. (2010). Usually ranges between 0.1-0.45.', attrs={"units": "Dimensionless"})#if deposit standard deviation were everywhere equal to the supply standard deviation, then there would be no fining of the mean sand size and C3 would be 0 (Fedele and Paola, 2007).
    porosity=xs.variable(dims=(),intent="in",description='Porosity', attrs={"units": "Dimensionless"})
    #Input/Outputs calculated in this class
    EFlux = xs.variable(dims=('y','x'),intent='out',description="Sediment flux calculated through an integration of erosion rate through the drainage stack", attrs={"units": "m/yr"}) #2D array
    R = xs.variable(dims=('y','x'),intent='out',description="Fedele and Paola (2007) Self-Similar Grain Size Calculation: R", attrs={"units": "Dimensionless"}) #2D array
    YStar = xs.variable(dims=('y','x'),intent='out',description="Fedele and Paola (2007) Self-Similar Grain Size Calculation: Y*", attrs={"units": "Dimensionless"}) #2D array
    DMean = xs.variable(dims=('y','x'),intent='out',encoding={'fill_value': None},description="D50 grain size fining computed for each time step", attrs={"units": "mm"}) #2D array
    D0Source = xs.variable(dims=('y','x'),intent='out',encoding={'fill_value': None}, description="D50 grain size fining surface updated over time (Also used for the calculation of DORG)", attrs={"units": "mm"}) #2D array ,static=True
    DTIME = xs.variable(dims=('y','x'),intent='out',encoding={'fill_value': None},description="D50 grain size at the source and propagated through the drainage network (Used for DMean calculation purposes)", attrs={"units": "mm"}) #2D array ,static=True
    Age = xs.variable(dims=('y','x'),intent='out',encoding={'fill_value': None},description="Age of deposition", attrs={"units": "yrs"}) #2D array ,static=True
    
    def initialize(self): 
        #Initialize all DMean, DTIME, D0Source grids with the starting bedrock (mountain catchement) grain size
        #D0T=np.full((self.gridshape),self.D0) 
        self.D0Source=np.full((self.gridshape),self.D0) 
        self.DTIME=np.full((self.gridshape),self.D0) 
        self.DMean=np.full((self.gridshape),self.D0) 
        self.Age=np.full((self.gridshape),0) 
        
    @xs.runtime(args=("step_end"))     
    def run_step(self,endtime): #Run_Step function only takes into accout at one time step
        
        #Sediment Flux
        flatE=self.E.flatten() #Flatten the 2D grid for computation through the 1D stack. Ravel is another version of flatten, but flatten is preferred. 
        EE=GSFast_Functions.sum_stack(self.stack,self.rec,flatE,self.nrec,self.rec_weights) #Calculate the erosional flux through the nodes in the stack
        flux = np.where(EE>0.00000000,EE,0.000000001) #Ensure that there are no negative flux values. 
        self.EFlux=np.reshape(flux,(self.gridshape)) #Reshape and save the calculated sediment flux as the EFlux output. 

        #Calculate Dimensionless R
        porosity=self.porosity
        RR=-(flatE/flux)*(1-porosity)#R is described in Felde and Paola (2007) as the deposition rate/ flux throughout the drainage network and is used to compute Y*.  
        RR=np.where(RR>0,RR,0)
        RR=np.where(EE>0,RR,0)
        self.R = np.reshape(RR,(self.gridshape))
        
        #Coefficients and Variables needed to compute grain size. #Described in Fedele and Paola (2007) with sample values in Duller et al. (2010).  
        D0= self.D0 #mm #Original bedrock source grain size 
        C3= self.C3 #C3 is a constant that describes the relation of the downstream change in the grain size standard deviation. 

        #Calculate ystar and the grain size source for the drainage network. 
        #Eg: The surface grain size at the head of the drainage network is taken from the previous DTIME grid and used as the source for the entire drainage network.
        #Flatten the necessary 2D grids to compute through the stack.
        D0T=self.DTIME.flatten()
        BedDepth=self.BedDepth.flatten()
        
        #compute ystar and Update the source grid
        D0T=np.where(BedDepth>0,D0T,D0)
        Y, D0Org=GSFast_Functions.qD0sum_stack(self.stack,self.rec,RR,flux,self.nrec,self.rec_weights,D0T) #R is combined through the stack to compute Y*. Fedele and Paola (2007) use an integration of Y* through the drainage network in the computation of grain size fining. 
        self.YStar=np.reshape(Y,(self.gridshape))
        D0Org=np.where(np.isnan(D0Org)==True, D0, D0Org)
        D0Org = D0Org.astype(float)
        self.D0Source=np.reshape(D0Org,(self.gridshape))
        
        #Calculate DMean grain size fining at this time step using the D0Source updated for this drainage time step .
        Dm=D0Org*(np.exp(-C3*Y))
        Dm=np.where(RR>0,Dm,np.nan)
        Dm=np.where(EE>0,Dm,np.nan)
        Dm = Dm.astype(float)
        self.DMean=np.reshape(Dm,(self.gridshape))
        
        #Add the DMean grain size fining to the previous surface of grain size deposits that are updated over time. 
        D0T=np.where(np.isnan(Dm)==True, D0T, Dm)
        self.DTIME=np.reshape(D0T,(self.gridshape))
        Age=self.Age.flatten()
        Age=np.where(np.isnan(Dm)==True, Age, endtime)
        self.Age=np.reshape(Age,(self.gridshape))
        # Here we recompute ystar assuming Rstar=1 to check that it creates a value comprised between 0 and 1
        R1Check=False
        if R1Check:
            RR=np.ones_like(RR)/(self.gridshape[1]-1)
            Y, D0Org=GSFast_Functions.qD0sum_stack(self.stack,self.rec,RR,flux,self.nrec,self.rec_weights,D0T) #R is combined through the stack to compute Y*. Fedele and Paola (2007) use an integration of Y* through the drainage network in the computation of grain size fining. 
            self.YStar=np.reshape(Y,(self.gridshape))

        #print(np.count_nonzero(np.isnan(self.D0ORG)))
        #print(np.count_nonzero(self.D0ORG==0))
        






