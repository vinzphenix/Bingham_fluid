from bingham_structure import *
from mosek import *


# Define a stream printer to grab output from MOSEK
def streamprinter(text):
    sys.stdout.write(text)
    sys.stdout.flush()


def func_append_var(numvar, task: mosek.Task):
    task.appendvars(numvar)

def solve_fem():
    inf = 0.

    with mosek.Task() as task:

        # Objective coefficients
        c = [3.0, 1.0, 5.0, 1.0]

        # Below is the sparse representation of the A
        # matrix stored by column.
        asub = [[0, 1],
                [0, 1, 2],
                [0, 1],
                [1, 2]]
        aval = [[3.0, 2.0],
                [1.0, 1.0, 2.0],
                [2.0, 3.0],
                [1.0, 3.0]]

        # Bound keys for constraints
        
        bkc = np.full(3, mosek.boundkey.fx)
        bkc[1] = mosek.boundkey.lo
        bkc[2] = mosek.boundkey.up
        
        # bkc = [mosek.boundkey.fx,
            #    mosek.boundkey.lo,
            #    mosek.boundkey.up]

        # Bound values for constraints
        blc = [30.0, 15.0, -inf]
        buc = [30.0, +inf, 25.0]

        # Bound keys for variables
        bkx = [mosek.boundkey.lo,
               mosek.boundkey.ra,
               mosek.boundkey.lo,
               mosek.boundkey.lo]

        # Bound values for variables
        blx = np.array([0.0, 0.0, 0.0, 0.0]) + 0
        bux = [+inf, 10.0, +inf, +inf]

        numvar = len(bkx)
        numcon = len(bkc)

        # Attach a log stream printer to the task
        task.set_Stream(mosek.streamtype.log, streamprinter)

        # Input the objective sense (minimize/maximize)
        task.putobjsense(mosek.objsense.minimize)

        # Append 'numcon' empty constraints.
        # The constraints will initially have no bounds.
        task.appendcons(numcon-1)

        # Append 'numvar' variables.
        # The variables will initially be fixed at zero (x=0).
        
        func_append_var(numvar, task)

        task.putclist(np.arange(numvar), -np.array(c))
        task.putaijlist(np.array([0, 0, 0, 1, 1, 1, 1]),
                        np.array([0, 1, 2, 0, 1, 2, 3]),
                        np.array([3., 1., 2., 2., 1., 3., 1.]))
        task.putconboundlist(np.arange(numcon-1), bkc[:-1], blc[:-1], buc[:-1])

        task.putvarboundlist(np.arange(numvar), bkx, blx, bux)

        task.appendcons(1)
        task.putaijlist(np.array([2, 2]),
                        np.array([1, 3]),
                        np.array([2., 3.]))
        task.putconboundlist([numcon-1], bkc[-1:], blc[-1:], buc[-1:])



        # Solve the problem
        task.optimize()

        # Print a summary containing information
        # about the solution for debugging purposes
        task.solutionsummary(mosek.streamtype.msg)

        # Get status information about the solution
        solsta = task.getsolsta(mosek.soltype.bas)

        if (solsta == mosek.solsta.optimal):
            xx = task.getxx(mosek.soltype.bas)
            print(xx)

        #     print("Optimal solution: ")
        #     for i in range(numvar):
        #         print("x[" + str(i) + "]=" + str(xx[i]))
        # elif (solsta == mosek.solsta.dual_infeas_cer or
        #       solsta == mosek.solsta.prim_infeas_cer):
        #     print("Primal or dual infeasibility certificate found.\n")
        # elif solsta == mosek.solsta.unknown:
        #     print("Unknown solution status")
        # else:
        #     print("Other solution status")


# call the main function
try:
    solve_fem()
except mosek.Error as e:
    print("ERROR: %s" % str(e.errno))
    if e.msg is not None:
        print("\t%s" % e.msg)
        sys.exit(1)
except:
    import traceback
    traceback.print_exc()
    sys.exit(1)
