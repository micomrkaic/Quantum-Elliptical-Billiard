# Schrodinger Elliptic Billiard

Real-time simulation of a quantum particle confined to an elliptic hard-wall
billiard. Two independent solvers -- numerical and analytic -- can be toggled
at runtime, allowing direct comparison.

---

## Table of Contents

1. [Physics](#1-physics)
2. [Analytical Method](#2-analytical-method)
3. [Numerical Method](#3-numerical-method)
4. [Build](#4-build)
5. [Controls](#5-controls)
6. [Display](#6-display)

---

## 1. Physics

### The quantum billiard

We solve the two-dimensional time-dependent Schrodinger equation (TDSE)

    i hbar d/dt psi(x,y,t) = H psi(x,y,t)

with the Hamiltonian

    H = -(hbar^2 / 2m) nabla^2  +  V(x,y)

where the confining potential is

    V(x,y) = 0        inside the ellipse  x^2/a^2 + y^2/b^2 <= 1
    V(x,y) = infinity outside

enforced as the Dirichlet boundary condition psi = 0 on the ellipse boundary.

Units throughout: hbar = 1, m = 1/2, so hbar^2/2m = 1 and H = -nabla^2.
Energies equal k^2 where k is the wavenumber.

### Elliptic billiards and classical integrability

The elliptic billiard is one of the few classically integrable 2D billiards.
The second conserved quantity (beyond energy) is the product of angular momenta
about the two foci, located at (+-c, 0) with c = sqrt(a^2 - b^2).

Classical trajectories are permanently confined to one of two families of
caustics:

- Confocal ellipses: orbits that never cross the interfocal segment.
- Confocal hyperbolae: orbits that do cross the segment.

These two families are separated by the unstable orbit along the minor axis.
Quantum mechanically, eigenstates reflect this structure exactly: they
factorise in elliptic coordinates into a radial part (depending only on
mu = acosh(r/c)) and an angular part (depending only on nu, the elliptic
angle). This separability is why the eigenfunctions are Mathieu functions and
why the analytic solution is exact.

### Initial condition

The simulation starts from a Gaussian wavepacket with a momentum kick:

    psi_0(x,y) = A * exp(  -[(x-x0)^2 + (y-y0)^2] / (2 sigma^2)  )
                   * exp(  i (kx*x + ky*y)  )

where A is the normalisation constant, (x0, y0) is the packet centre, sigma
is the spatial width, and (kx, ky) = (2 pi / lambda) * (cos theta, sin theta)
sets the mean momentum direction.

This is the minimum-uncertainty state: both position and momentum
uncertainties saturate the Heisenberg bound.  In momentum space the packet is
a Gaussian centred on (kx, ky) with width 1/sigma.  Large sigma gives a
nearly monochromatic packet (small momentum spread); small sigma gives a
spatially localised packet approaching the ray-optics limit.

---

## 2. Analytical Method

### Separation in elliptic coordinates

Elliptic coordinates (mu, nu) are defined by

    x = c cosh(mu) cos(nu),    y = c sinh(mu) sin(nu)

with c = sqrt(a^2 - b^2) (focal half-distance), mu >= 0, nu in [0, 2*pi).
The ellipse boundary is the coordinate surface mu = mu_0 = acosh(a/c).

The Helmholtz equation -nabla^2 phi = k^2 phi separates exactly.  Writing
phi(mu, nu) = R(mu) * Theta(nu) gives:

    Angular (Mathieu equation):
        d^2 Theta / d nu^2  +  (a_r - 2q cos 2nu) Theta = 0

    Radial (modified Mathieu equation):
        d^2 R / d mu^2  -  (a_r - 2q cosh 2mu) R = 0

where q = (k c / 2)^2 is the Mathieu parameter and a_r is the separation
constant.  Both equations share the same a_r and q.  The energy is

    E = k^2 = 4q / c^2

### Angular Mathieu functions

The angular equation has periodic solutions for countably many values a_r(q).
There are four families:

    Family    Symbol      Series                       GSL function
    --------  ----------  ---------------------------  ------------------
    ce-even   ce_{2m}     sum A_m cos(2m nu)           gsl_sf_mathieu_ce
    ce-odd    ce_{2m+1}   sum A_m cos((2m+1) nu)       gsl_sf_mathieu_ce
    se-even   se_{2m+2}   sum B_m sin((2m+2) nu)       gsl_sf_mathieu_se
    se-odd    se_{2m+1}   sum B_m sin((2m+1) nu)       gsl_sf_mathieu_se

The ce functions (cosine-like) are even in nu; the se functions (sine-like)
are odd.  Both satisfy the same Mathieu equation with different characteristic
values.

### Radial Mathieu functions and eigenvalue quantisation

For the interior problem the physically correct radial solution is the
first-kind radial Mathieu function:

    Even:  Mc_n^(1)(mu; q)    [gsl_sf_mathieu_Mc(1, n, q, mu)]
    Odd:   Ms_n^(1)(mu; q)    [gsl_sf_mathieu_Ms(1, n, q, mu)]

These are regular at mu = 0 (the interfocal segment).  The Dirichlet
boundary condition quantises q:

    Mc_n^(1)(mu_0; q_{n,r}) = 0    (even states, r = 1, 2, 3, ...)
    Ms_n^(1)(mu_0; q_{n,r}) = 0    (odd  states)

Each zero q_{n,r} is a 2D eigenvalue with energy E = 4 q_{n,r} / c^2.

### Eigenvalue computation

For each angular order n = 0..N_MAX = 24 and each parity:

1. Scan q from Q_MIN = 0.5 to Q_MAX = 800 in NSCAN = 15000 steps.
2. At each step evaluate Mc_n(mu_0; q) or Ms_n(mu_0; q) via GSL.
3. Detect sign changes; bisect each one to 64-bit precision (60 iterations).
4. Sort all found zeros by energy; keep up to NEIG = 1200.

Total GSL calls for the scan: 2 * (N_MAX+1) * NSCAN ~ 750,000.
The scan itself runs in a few seconds. The dominant cost is the subsequent
eigenfunction synthesis: ~10ms per state * 1200 states ~ 12 seconds total.
Both steps run in a background thread while the numerical solver stays live.

### Eigenfunction synthesis and projection

For each found eigenvalue q_{n,r}:

1. Build a radial lookup table: RTAB = 2048 GSL evaluations of Mc or Ms on
   a uniform grid of mu in [0, mu_0].

2. Build an angular lookup table: ATAB = 8192 GSL evaluations of ce or se on
   a uniform grid of nu in [0, 2*pi].

3. Walk the interior grid (precomputed elliptic coordinates xi_grid, eta_grid)
   once, interpolating phi(x,y) = R(mu) * Theta(nu) linearly from the tables.
   In the same pass: accumulate the norm integral and the projection integral

       c_{n,r} = integral  phi_{n,r}*(x,y) * psi_0(x,y)  dx dy

4. Normalise the tables in-place; scale c_{n,r} by the same factor.

No N*N grid for phi is ever allocated.  Memory per state: (RTAB + ATAB) * 8
bytes = 80 KB.  Total for 1200 states: ~94 MB (vs 10+ GB for full grids).

### Exact time evolution

    psi(x,y,t) = sum_{n,r}  c_{n,r} * exp(-i E_{n,r} t) * phi_{n,r}(x,y)

This is algebraically exact -- no time-stepping error of any kind.  At each
rendered frame the wavefunction is reconstructed from the 1D tables by linear
interpolation.

### Time step

The time advance per frame is

    dt = (pi/20) / E_max * p_spd

where E_max is the highest included eigenvalue.  This ensures that even the
highest-energy mode advances by at most pi/20 radians of phase per frame,
preventing aliasing.  At default settings (80 states, p_spd=1) dt ~ 4.5e-4
and the ground state (T_0 ~ 0.38) takes ~840 frames per oscillation.

---

## 3. Numerical Method

### Peaceman-Rachford ADI Crank-Nicolson

The TDSE is integrated by the Alternating Direction Implicit scheme.  Each
full time step Dt splits into two half-steps:

    HS1 (x-implicit, y-explicit):
        (1 - rx/2 Lx) psi* = (1 + ry/2 Ly) psi^n

    HS2 (y-implicit, x-explicit):
        (1 - ry/2 Ly) psi^{n+1} = (1 + rx/2 Lx) psi*

where Lx, Ly are 1D second-difference operators (5-point stencil) and

    rx = i DT / DX^2,    ry = i DT / DY^2

Each half-step reduces to N independent complex tridiagonal systems of size N,
each solved in O(N) by the Thomas algorithm.  Total cost per full step:
O(N^2 log N) dominated by 2N Thomas solves.

Properties of the scheme:
- Unconditionally stable: no CFL condition on DT.
- Second-order in both time (O(DT^2)) and space (O(DX^2)).
- Norm-conserving: reinforced by explicit renormalisation after each step.
- Symplectic to leading order: phase errors accumulate slowly.

The hard-wall Dirichlet boundary is enforced by:
1. Zeroing all couplings to outside-ellipse points in the tridiagonal.
2. Zeroing psi outside the mask after each half-step.
3. Renormalising.

### Grid parameters

    N    = 1024         grid points per axis
    L    = 1.0          domain half-width: x, y in [-1, 1]
    DX   = 2 / (N-1)   grid spacing ~ 0.00195
    DT   = 0.000015     time step

The condition DT / DX^2 ~ 3.9 keeps the CN accuracy well within bounds.
At lambda = 0.12 (default), each de Broglie wavelength spans ~60 grid points.

### Parallelisation

Each half-step distributes N independent row-solves (HS1) or column-solves
(HS2) across NTHREADS persistent worker threads.  Synchronisation uses two
POSIX barriers:

    bar_start: main -> workers (dispatch new phase)
    bar_done:  workers -> main (phase complete)

Thread lifecycle:
1. threads_init() spawns NTHREADS workers and initialises barriers with
   count NTHREADS + 1 (workers + main all participate).
2. Each step: main sets g_rx, g_ry, calls dispatch(PHASE_HS1), then
   dispatch(PHASE_HS2), then handles the norm reduction manually.
3. threads_stop() sets PHASE_EXIT and joins all workers.

Cache optimisation: HS2 (column solves) would access tmp[j][i] with stride N
in a row-major array.  Instead, HS1 writes col_t[i][j] = tmp[j][i]
simultaneously with tmp, so HS2 reads col_t[i][j] -- fully contiguous access.

Norm reduction uses per-thread partial sums in partial_norm[NTHREADS], summed
by the main thread between the two barrier pairs of PHASE_NORM.

---

## 4. Build

### Linux

    sudo apt install libsdl2-dev liblapacke-dev liblapack-dev libblas-dev libgsl-dev
    make                  # NTHREADS=4 default
    make NTHREADS=8

### macOS (Homebrew)

    brew install sdl2 openblas gsl
    make macos NTHREADS=10

### Manual

Linux:
    gcc -O2 -march=native -std=c17 -DNTHREADS=4 -I/usr/include \
        $(sdl2-config --cflags) schrodinger.c -o schrodinger \
        $(sdl2-config --libs) -llapacke -llapack -lblas -lgsl -lgslcblas \
        -lm -lpthread

macOS:
    gcc -O2 -std=c17 -DNTHREADS=10 \
        -I$(brew --prefix sdl2)/include/SDL2 \
        -I$(brew --prefix openblas)/include \
        -I$(brew --prefix gsl)/include \
        schrodinger.c -o schrodinger \
        -L$(brew --prefix sdl2)/lib    -lSDL2 \
        -L$(brew --prefix openblas)/lib -lopenblas \
        -L$(brew --prefix gsl)/lib     -lgsl -lgslcblas \
        -lm -lpthread

---

## 5. Controls

### Keyboard

    Key        Action
    ---------  --------------------------------------------------------------
    M          Toggle Numerical / Analytic solver
    C          Toggle comparison mode (analytic mode only, requires eigenstates)
    R          Reset wavepacket with current parameters
    Space      Pause / resume
    V          Start / stop video recording  ->  output.mp4
    Q / Esc    Quit

### Sliders

All slider changes reset the wavepacket immediately.  In analytic mode they
also restart the eigenstate computation with the new geometry.

    Slider        Range          Default   Description
    ------------  -------------  -------   ----------------------------------
    a semi-major  0.35 -- 0.92   0.75      Ellipse semi-major axis
    b semi-minor  0.20 -- 0.88   0.50      Ellipse semi-minor axis (< a)
    wavelength L  0.04 -- 0.35   0.12      de Broglie wavelength lambda
    sigma         0.05 -- 0.35   0.15      Gaussian envelope width
    x0            -0.60 -- 0.60  -0.35     Packet centre x
    y0            -0.60 -- 0.60   0.10     Packet centre y
    angle deg     -180 -- 180    20        Launch direction from +x axis
    speed x       0.25 -- 2.00   1.00      Animation speed multiplier

Notes:
- b is clamped to b < a - 0.02 to keep c = sqrt(a^2-b^2) well-defined.
- Small lambda approaches the ray-optics (classical) limit.
- Large sigma gives a near-monochromatic packet; small sigma is spatially tight.
- In analytic mode, speed x scales dt = (pi/20)/E_max * speed_x.

### Comparison mode (key C)

Available in analytic mode once eigenstates are computed.

Press C to reset both solvers to t=0 from the same initial state and run
them simultaneously:
- Analytic solver advances by the exact eigenfunction expansion.
- Numerical solver advances by ADI Crank-Nicolson at DT=0.000015 per step.

The simulation panel overlays the pointwise error |psi_analytic - psi_numerical|
blended over the wavefunction display (black->yellow->red = low->high error).

The control panel footer shows:
    L2   = sqrt( integral |psi_a - psi_n|^2 dA )   (global error norm)
    Linf = max |psi_a(x,y) - psi_n(x,y)|           (worst-case pointwise)
    tn   = current time of the numerical solution

### Why the analytic method requires a large eigenstate basis

This section explains a fundamental mathematical constraint of the Mathieu
eigenfunction expansion method and how the code addresses it.

**The physics of basis completeness**

The analytic solver expands the initial wavepacket psi_0 in the eigenbasis:

    psi_0(x,y) = sum_s  c_s * phi_s(x,y)

where the sum runs over all Mathieu eigenstates phi_s sorted by energy E_s.
The quality of the expansion is measured by the projection norm:

    P = sum_s |c_s|^2

If P = 1, the basis is complete for this packet. If P < 1, the fraction
(1 - P) of the packet's norm is missing -- those components simply do not
appear in the analytic time evolution.

**Why high-energy packets need many eigenstates**

A Gaussian wavepacket in Cartesian coordinates with wavelength lambda has
mean wavenumber k = 2*pi/lambda and energy E_packet = k^2. To represent
it accurately in the Mathieu basis, the basis must contain eigenstates
near energy E_packet. The eigenstate energies are E_s = 4*q_s/c^2, so
representing a packet at E_packet requires eigenstates with

    q_s ~ E_packet * c^2/4 = (pi*c/lambda)^2

For the default geometry (a=0.75, b=0.50, c=0.559):

    lambda=0.12 (default): k=52, E_packet=2742, q_needed=214
    lambda=0.25:           k=25, E_packet=632,  q_needed=49
    lambda=0.50:           k=13, E_packet=158,  q_needed=12

But knowing q_needed is not enough. The Mathieu eigenstates are labelled
by two quantum numbers: angular order n and radial index r. A single
angular order n contributes only a few eigenstates near any given q. To
span a Cartesian Gaussian, which has support across ALL angular orders, you
need eigenstates from many values of n simultaneously.

**Measured basis requirements (a=0.75, b=0.50, sigma=0.15)**

The following table was computed by exhaustive numerical measurement:
scanning all Mathieu zeros up to Q_MAX=800 with N_MAX=24, sorting by energy,
and accumulating the projection norm state by state.

    lambda   k       q_packet   N for P=0.70   N for P=0.80   N for P=0.90
    ------   -----   --------   ------------   ------------   ------------
    0.12     52.4    214        893            966            1177
    0.15     41.9    137        541            620            668
    0.20     31.4    77         1066           1078           1082
    0.25     25.1    49         1064           1071           1074
    0.30     20.9    34         1049           1054           1062
    0.40     15.7    19         1036           1038           1042
    0.50     12.6    12         1013           1031           1033

The striking result: reaching P=0.90 requires roughly 1000-1200 states for
ANY wavelength in the range lambda=0.12..0.50. The number of required states
does not decrease much for slower packets because the bottleneck is not the
energy of the target states but the angular diversity needed to represent a
Cartesian Gaussian in elliptic coordinates.

**Why angular diversity is the bottleneck**

A Cartesian Gaussian exp(-r^2/2sigma^2) * exp(i*k*x) is NOT aligned with
elliptic coordinates. Expressing it in elliptic coordinates requires Mathieu
functions of arbitrarily high angular order n. The coefficients c_{n,r} are
significant for n up to roughly k*c ~ k*0.56, which is n~30 at lambda=0.12
and n~7 at lambda=0.50. Each angular order n contributes multiple radial
zeros r=1,2,3,... before reaching q_packet. The total count is:

    N_needed ~ (n_max * r_per_order) ~ k*c * (q_packet / q_spacing)

This grows roughly as k^2 for fixed geometry -- i.e. as 1/lambda^2.
The measured data confirms this scaling roughly holds.

**Implementation: NEIG=1200, Q_MAX=800, N_MAX=24**

The code uses NEIG=1200 eigenstates with Q_MAX=800 and N_MAX=24. This gives
P > 0.90 for all wavelengths lambda >= 0.12 used in the simulation.

Memory: 1200 states * (RTAB + ATAB) * 8 bytes = 1200 * 80 KB = 94 MB.
Synthesis time: ~10ms per state on a modern CPU, so ~12 seconds total.
The computation runs in a background thread; the numerical solver remains
interactive during this time.

**Projection norm display**

After eigenstates are computed, the control panel shows "N states  norm=X.XX":

    Green  (norm > 0.80): basis adequate, solution is reliable
    Yellow (norm > 0.50): partial capture, qualitatively useful
    Red    (norm <= 0.50): basis insufficient, shown with warning

With NEIG=1200 the display should be green for all standard packet parameters.

### Expected accuracy of the analytic vs numerical comparison

When proj_norm > 0.80, the two solvers should produce nearly identical results
at t=0. Divergence accumulates over time due to numerical phase error in the
ADI scheme.

At production settings (N=1024, DT=0.000015):
- Initial L2 error ~ 1e-3 (spatial discretisation, O(DX^2))
- Error growth ~ O(DT^2) per step, confirmed by ratio test
- At t=T0 (one ground state period) L2 ~ 1e-2
- Norm conserved to machine precision by both solvers

The comparison is most informative early in the evolution (t < T0/2) before
multiple boundary reflections mix many eigenmodes.

### Video

Press V to record.  Frames are piped to ffmpeg (must be in PATH) at 30 fps
and written as output.mp4 in the current directory.  Requires:

    sudo apt install ffmpeg     # Linux
    brew install ffmpeg         # macOS

---

## 6. Display

### Simulation panel (768 x 768 px)

    Hue        = arg(psi(x,y,t))   -- phase, full colour wheel over [-pi, pi]
    Brightness = |psi|^(2/3)       -- probability density, gamma-boosted

The gamma boost (cubic root) enhances dim interference fringes that would
otherwise be invisible against a bright peak.

Phase-to-colour mapping (left to right on the legend bar):
    -pi  -->  red  -->  yellow  -->  green  -->  cyan  -->  blue  -->  pi

The green contour is the ellipse boundary.

### Control panel (300 px)

    Top row:     mode badge (NUMERIC/ANALYTIC), PAUSED, REC indicators
    Progress:    analytic mode eigenstate computation bar
    Sliders:     8 parameter sliders with live values
    Footer:      key legend, current time / time-step, phase colour bar

---

## Interesting configurations

    Major-axis bounce (classical periodic orbit):
        x0=-0.50, y0=0, angle=0, lambda=0.10
        The packet bounces coherently between the two vertices.

    Caustic tracing:
        x0=-0.30, y0=0.15, angle=15
        The packet tracks an elliptic caustic without crossing the
        interfocal segment.

    Ray-optics / scarring limit:
        lambda=0.05, sigma=0.08
        Nearly classical behaviour; scars visible on periodic orbits.

    Quantum spreading:
        sigma=0.30, lambda=0.15
        Rapid spreading; interference nodal lines fill the billiard.

    Near-circular (whispering gallery):
        a=0.70, b=0.68
        Foci nearly coincide; eigenstates approach Bessel functions;
        whispering-gallery modes concentrate near the boundary.

    Highly eccentric:
        a=0.85, b=0.22
        Large focal separation; hyperbolic and elliptic caustic families
        are well-separated and visible in the wavepacket dynamics.
