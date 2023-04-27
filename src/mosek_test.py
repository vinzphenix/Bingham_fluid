from bingham_structure import *
from mosek import *


# Define a stream printer to grab output from MOSEK
def streamprinter(text):
    sys.stdout.write(text)
    sys.stdout.flush()


def func_append_var(numvar, task: mosek.Task):
    task.appendvars(numvar)

def linear_pb():
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


def conic_pb():
    n, k = 3, 2
    inf = 0.
    # Create a task
    with mosek.Task() as task:
        # Attach a printer to the task
        task.set_Stream(mosek.streamtype.log, streamprinter)

        # Create n free variables
        task.appendvars(n)
        task.putvarboundsliceconst(0, n, mosek.boundkey.fr, -inf, inf)

        # Set up the objective
        c = [1, 1, 0]
        task.putobjsense(mosek.objsense.minimize)
        task.putclist(range(n), c)

        # One linear constraint - sum(x) = 1
        # task.appendcons(1)
        # task.putarow(0, range(n), [1]*n)
        # task.putconbound(0, mosek.boundkey.fx, 1.0, 1.0)


        # Append empty AFE rows for affine expression storage
        task.appendafes((k + 1)+(k+1))

        # G matrix in sparse form
        Fsubi = [0, 0, 1, 1, 2, 2]
        Fsubj = [0, 1, 0, 1, 0, 2]
        Fval  = [-.5, -1., 3., -.5, 1., -2.]
        g     = [20., -15., -10.]

        # Fsubi_ = np.array([0, 0, 1, 2, 2, 3, 3])+3
        # Fsubj_ = [0, 1, 2, 0, 1, 1, 2]
        # Fval_  = [1., -.5, 1., -.25, 2., 1., -3.]
        # g_     = [0., 0., -5., +10.]

        Fsubi_ = np.array([0, 0, 1, 1, 2, 2])+3
        Fsubj_ = [0, 1, 0, 1, 1, 2]
        Fval_  = [1., -.5, -.25, 2., 1., -3.]
        g_     = [0., -5., +10.]

        # Fill in F storage
        task.putafefentrylist(Fsubi, Fsubj, Fval)
        task.putafefentrylist(Fsubi_, Fsubj_, Fval_)
        print(spmatrix(Fval, Fsubi, Fsubj))

        # Fill in g storage
        task.putafegslice(0, k+1, g)
        task.putafegslice(k+1, 2*k+2, g_)

        # Define a conic quadratic domain
        quadDom = task.appendquadraticconedomain(k + 1)
        rquadDom = task.appendrquadraticconedomain(k + 2)

        # Create the ACC
        print(np.arange(2*k+2))
        task.appendaccs([quadDom]*2, np.arange(2*k+2), None)
        # task.appendaccs([], np.array([]), None)

        # task.appendacc(quadDom,    # Domain index
        #                range(k+1), # Indices of AFE rows [0,...,k]
        #                None)       # Ignored
        # task.appendacc(quadDom,    # Domain index
        #                range(k+1, 2*k+2), # Indices of AFE rows [0,...,k]
        #                None)       # Ignored

        # task.appendacc(rquadDom,    # Domain index
        #                range(k+1, 2*k+3), # Indices of AFE rows [0,...,k]
        #                None)       # Ignored

        # Solve and retrieve solution
        task.optimize()
        xx = task.getxx(mosek.soltype.itr)
        if task.getsolsta(mosek.soltype.itr) == mosek.solsta.optimal:
            print("Solution: {xx}".format(xx=list(xx)))
    return


# call the main function
try:
    conic_pb()
except mosek.Error as e:
    print("ERROR: %s" % str(e.errno))
    if e.msg is not None:
        print("\t%s" % e.msg)
        sys.exit(1)
except:
    import traceback
    traceback.print_exc()
    sys.exit(1)
