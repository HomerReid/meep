--8<-- "doc/docs/AdjointSolver/AdjointDocumentationStyleHeader.md"

---
# Design optimization with the <span class=SC>meep</span> adjoint solver: A reference manual
---

As described in the [Overview](Overview.md), the first step in
using `mp.adjoint` is to write a python script implementing
a subclass of the `OptimizationProblem` abstract base class
defined by `mp.adjoint`. Once that's ready, you can run
your script with command-line options instructing `mp.adjoint`
to carry out various calculations; these may be *single-point*
calculations, in which the geometry is fixed at given set
of values you specify for the design variables (thus defining
a single point in the space of possible input), or full-blown
*iterative optimizations* in which `mp.adjoint` automatically
evolves the design variables toward the values that optimize
your objective. 

> :bookmark:{.summary} **<span class=SC>table of contents</span>**
>
> 1. **Defining your problem**: Writing a python script for `mp.adjoint`
> 
>     a. The `OptimizationProblem` abstract base class
>
>     b. Mandatory class-method overrides: `init_problem` and `create_sim`
>
>     c. Optional class-method override: `add_args`
>
> 2. **Running your problem**: Visualization, single-point calculations, and full iterative optimization
>
>     a. Visualizing your geometry
>
>     b. Evaluating the objective function value and gradient at a single design point
>
>     c. Testing gradient components by finite-differencing
>
>     d. Running many single-point calculations in parallel:`ParallelDesignTester`
 
>     e. Full iterative optimization
>
> [4. Built-in command-line options](#4-built-in-command-line-options)

## 1. Defining your problem: Writing a python script for `mp.adjoint`

### 1a. The `OptimizationProblem` abstract base class

As described in the [Overview](Overview.md), the python script
that drives your `mp.adjoint` session implements a subclass
of the `OptimizationProblem` base class defined by `mp.adjoint.`
This is a high-level abstraction of the design-automation 
process; the base class knows how to do various general things
involving <span class=SC>meep</span> geometries and objective
functions, but is lacking crucial information from you, 
without which it can't do anything on its own.
That is to say, `OptimizationProblem` is an
[abstract base class](https://docs.python.org/3/library/abc.html)
with two pure virtual methods that your derived class
must override to describe the specifics of your design problem.

### 1b. Mandatory class-method overrides

More specifically, your subclass of `OptimizationProblem`
must furnish implementations of the following two pure virtual
methods left unimplemented in the base class. (Click the header
bars to unfold the description of each item.)


??? summary "`init_problem`: One-time initialization"
    Your `init_problem` routine will be called once, at the beginning of a `mp.adjoint`
    session; think of it as the class constructor. (Indeed, it is called from the
    parent class constructor.) 
    It has two purposes: **(a)** to give you a chance to complete any one-time initialization
    tasks you need to do, and **(b)** to communication to the base class the
    [CommonElementsOfOptimizationGeometries](Overview.md#common-elements-of-optimization-geometries-objective-regions-objective-functions-design-regions-basis-sets)
    needed to turn a generic <span class=SC>meep</span> simulation into an optimization problem.

    - **Calling convention:** `def init_problem(self,args)`

        - `args`: Structure containing values of all command-line options.

    - **Return values:**

        The routine should return a 5-tuple

       ```py3 
           fstr, objective_regions, extra_regions, design_region, basis
       ```

    where

    - `fstr` is a string specifying your objective function

    - `objective_regions` is a list of regions over which to compute
           frequency-domain (DFT) fields needed to evaluate the quantities
           on which your objective function depends

    -  `extra_regions` is a list of additional regions over which to compute
           DFT fields for post-processing or visualization; it will often be
           just the empty list `[]`

    - `design_region` is a region encompassing the variable-permittivity
           region of your geometry

    - `basis` is a specification of the set of basis functions used to 
           expand the permittivity. This is a subclass of the `Basis` base class
           defined by `meep.adjoint`; you may implement your own arbitrary
           basis, or use one of several predefined bases provided by `meep.adjoint.`

---

??? summary "`create_sim`: Instantiating a geometry with given design variables"
    Your `create_sim` routine will be called each time `mp.adjoint` needs to
    compute your objective function for a particular set of design-variable
    values.

--------------------------------------------------

### 1c. Optional class-method override

??? summary "`add_args`: Configure problem-specific command-line arguments"
    The `OptimizationProblem` base class defines a
    [large number of general-purpose command-line options](#4-built-in-command-line-options),
    with judiciously chosen default values, to control `mp.adjoint` calculations. 
    In many cases you will want **(1)** to reconfigure the default values as
    appropriate for your problem, **(2)** to add additional options relevant to
    your specific geometry. Your subclass can do both of these things by overriding
    the `add_args` class method, which will be called just before the actual
    command-line arguments are parsed.

     **Prototype:**  `init_args(self,parser)`

     - `parser`: [`argparse`](https://docs.python.org/3/library/argparse.html)

         A structure that has been initialized with
         [all built-in `mp.adjoint` options](#4-built-in-command-line-options)
         and their default values.
         You may call `parser.set_defaults` to change the default values of
         built-in options and `parser.add_argument` to add new options.
         (The actual values specified for all arguments are made available
         to you via the `args` parameter passed to your `init_problem`
         routine.)

     **Return values:**  None.

--------------------------------------------------

### Objective-function specifications

Your objective function is specified by the string `fstr` that you return as
the first element in the 5-tuple return value of `init_problem.`

Your objective function will depend on one or more objective quantities,
such as power fluxes or eigenmode expansion coefficients, associated
with specific objective regions in your geometry. `mp.adjoint` defines
a convention for assigning a unique character string to each 
objective function in your geometry. Your `fstr` should use
these labels to refer to the objective quantities on which it
depends.

>:bookmark: **Naming convention for objective quantities**
>
> | Quantity                                                                         |  Label      |
  |:--------------------------------------------------------------------------------:|:------------|
  |  Power flux through flux region `r`                                              | `S_r`       |
  |  Expansion coefficient for forward-travelling eigenmode *n* at flux region `r`   | `Pn_r`      |
  |  Expansion coefficient for backward-travelling eigenmode *n* at flux region `r`  | `Mn_r`      |
>
> Note that the flux-region label `r` may be a character string like `east` if your `init_problem` 
  method assigned a name to the `DFTCell` for that region. Otherwise, `r` is an integer-valued
  index corresponding to the zero-based index of the DFT cell in the `objective_regions` list
  returned by your `init_problem`.

### Expansion bases

The `basis` field returned by `init_problem` is a subclass of the 
`Basis` base class implemented by `meep.adjoint.` You can write 
your own subclass to define an arbitrary basis, or use one of the 
built-in basis sets provided by `meep.adjoint.` A good default
choice is `FiniteElementBasis`: 

```py3
   basis = FiniteElementBasis(lx=4, ly=4, density=4)
```

which defines a basis of localized
functions over a rectangle of dimensions `l_x`&times;`l_y` with `density`
elements per unit length.

## 2. **Running your problem**: Visualization, single-point calculations, and full iterative optimization

Having implemented your script, you can execute it as a python script to
run various calculations specified by command-line options.

### 2a. Visualizing your geometry

### 2b. Evaluating the objective function value and gradient at a single design point

### 2c.  Testing gradient components by finite-differencing

### 2d. Running many single-point calculations in parallel: `ParallelDesignTester`

### 2e. Full iterative optimization

### 2f. Running many single-point calculations in parallel: `ParallelDesignTester`


## 4. Built-in command-line options

The following command-line options are defined by the `OptimizationProblem`
base class and are available in all `mp.adjoint` sessions.

### Options affecting <span class=SC>meep</span> timestepping

| Option                            | Description                                                   |
| --------------------------------- | --------------                                                |
| `#!py3 --res`                     | resolution
| `#!py3 --dpml`                    | PML thickness (-1 --> autodetermined)
| `#!py3 --fcen`                    | center frequency
| `#!py3 --df`                      | frequency width
| `#!py3 --source_mode`             | mode index of eigenmode source
| `#!py3 --dft_reltol`              | convergence threshold for end of timestepping
| `#!py3 --dft_timeout`             | max runtime in units of last_source_time
| `#!py3 --dft_interval`            | meep time DFT convergence checks in units of last_source_time

### Options affecting outputs from <span class=SC>meep</span> computations

| Option                            | Description                                                    |
| --------------------------------- | --------------                                                 |
| `#!py3 --nfreq`                   | number of output frequencies
| `#!py3 --full_dfts`               | compute DFT fields over full volume
| `#!py3 --complex_fields`          | force complex fields
| `#!py3 --filebase`                | base name of output files

### Options specifying initial values for basis-function coefficients

| Option                            | Description                                                   |
| --------------------------------- | --------------                                                |
| `#!py3 --betafile`                | file of expansion coefficients
| `#!py3 --beta`                    | set value of expansion coefficient
| `#!py3 --eps_design`              | functional expression for initial design permittivity

### Options describing the calculation to be done

| Option                            | Description                                                   |
| --------------------------------- | --------------                                                |
| `#!py3 --eval_objective`          | evaluate objective function value
| `#!py3 --eval_gradient`           | evaluate objective function value and gradient
| `#!py3 --gradient_qname`          | name of objective quantity to differentiate via adjoint method
| `#!py3 --fd_order`                | finite-difference order (0,1,2)
| `#!py3 --fd_index`                | index of differentiation variable
| `#!py3 --fd_rel_delta`            | relative finite-difference delta
| `#!py3 --optimize`                | perform automated design optimization

### Options affecting optimization

| Option                            | Description                                                   |
| --------------------------------- | --------------                                                |
| `#!py3 --alpha`                   | gradient descent relaxation parameter
| `#!py3 --min_alpha`               | minimum value of alpha
| `#!py3 --max_alpha`               | maximum value of alpha
| `#!py3 --boldness`                | sometimes you just gotta live a little
| `#!py3 --timidity`                | can\'t be too careful in this dangerous world
| `#!py3 --max_iters`               | max number of optimization iterations

### Options configurating adjoint-solver options

| Option                            | Description                                                   |
| --------------------------------- | --------------                                                |
| `#!py3 --verbose`                 | produce more output
| `#!py3 --concise`                 | produce less output
| `#!py3 --visualize`               | produce visualization graphics
| `#!py3 --label_source_regions`    | label source regions in visualization plots
| `#!py3 --logfile`                 | log file name
| `#!py3 --pickle_data`             | save state to binary data file
| `#!py3 --animate_component`       | plot time-domain field component
| `#!py3 --animate_interval`        | meep time between animation frames

--8<-- "doc/docs/AdjointSolver/AdjointLinks.md"
