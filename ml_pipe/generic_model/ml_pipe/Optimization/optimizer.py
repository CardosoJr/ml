import pandas as pd
import numpy as np
import pyomo.environ as pyo
import pathlib
import zipfile
import os
import traceback

class Optimizer: 
    def __init__(self, lock, optimizer_name = 'gurobi', optimizer_path = ''):
        self.__lock = lock
        
        if optimizer_name == 'gurobi':
            # set env variables
            os.environ['GUROBI_HOME'] = "/opt/gurobi900/linux64"
            os.environ['PATH'] =  os.environ['PATH'] + ":" + os.environ['GUROBI_HOME'] + "/bin"
            
            if 'LD_LIBRARY_PATH' in os.environ.keys():
                os.environ['LD_LIBRARY_PATH'] = os.environ['LD_LIBRARY_PATH'] + ":" + os.environ['GUROBI_HOME'] + "/lib"
            else:
                os.environ['LD_LIBRARY_PATH'] = os.environ['GUROBI_HOME'] + "/lib"
                
            self.opt = pyo.SolverFactory(optimizer_name, solver_io = 'python')
            
        else:
            self.opt = pyo.SolverFactory(optimizer_name, optimizer_path)
            
    def optimize(self, df_curvas_elasticidade, min_global_conversion,  interval = 0.01):
        self.__lock.acquire()
        try:
            df_curvas_elasticidade, obj_value, global_conversion, results, model = self.__optimize(df_curvas_elasticidade, min_global_conversion, interval)
        except Exception as e:
            print('error while optimizing')
            print(str(e))
            print(traceback.format_exc())
        finally: 
            self.__lock.release()
            
        return  df_curvas_elasticidade, obj_value, global_conversion, results, model
            
    def __optimize(self, df_curvas_elasticidade, min_global_conversion,  interval = 0.01):
        # N users
        n_users = df_curvas_elasticidade['u_id'].nunique()

        # Initialize model
        model = pyo.ConcreteModel()

        # Decision variables
        model.x = pyo.Var(((c_id,a) for c_id, a in df_curvas_elasticidade[['u_id','ajuste']].values), within=pyo.Binary)

        # Objective: Maximize margin 
        e = sum((model.x[u_id, ajuste] * obj) for (u_id, ajuste, obj) in [tuple(x) for x in  df_curvas_elasticidade[['u_id','ajuste','obj']].values])

        model.obj = pyo.Objective(expr = e, sense=pyo.maximize)
        
        # Constraints ---
        model.constraints = pyo.ConstraintList()

    #     # Must pick one for each id
        for c_id, g in df_curvas_elasticidade.groupby('u_id'):
            model.constraints.add(1 == sum([model.x[c_id, a] for a in g['ajuste'].values]))

        # Must have a global minumum conversion rate
        try:
            threshold = n_users * min_global_conversion
            current = sum([model.x[c_id, a] * prob_renov for c_id, a, prob_renov in df_curvas_elasticidade[['u_id','ajuste','prob_renov']].values])
            model.constraints.add(current - threshold >= 0)
            
        except: 
            print('ERRO GLOBAL CONVERSION', n_users * min_global_conversion)
            raise

#         model.pprint()
        results = self.opt.solve(model)
        
        # Get results
        # Print the status of the solved LP
        print("Status = %s" % results.solver.termination_condition)
        df_result = df_curvas_elasticidade.copy(deep = True)

        if str(results.solver.termination_condition) == 'optimal':
            # Store values for decision variables into the DF
            df_result['x_ij'] = [pyo.value(model.x[c_id, a]) for c_id, a in df_result[['u_id','ajuste']].values]

            # Value objective function
            obj_value = (df_result['obj']*df_result['x_ij']).sum()
            print('Obj. value: ', obj_value)

            # Value conversion rate
            global_conversion = (df_result['prob_renov']*df_result['x_ij']).sum()/n_users
            print('Global expected conversion: ', global_conversion)
        else:
            obj_value = np.nan
            global_conversion = np.nan

        # Include params
        df_result['min_global_conversion'] = min_global_conversion

        # return
        return  df_result, obj_value, global_conversion, results, model