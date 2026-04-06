/*
 * schrodinger.c = 2D TDSE, elliptic hard-wall billiard
 *
 * Modes (M to toggle):
 *   NUMERICAL = parallel ADI Crank-Nicolson
 *   ANALYTIC  = Mathieu eigenfunction expansion, exact time evolution
 *               Uses LAPACKE dsyev on small (~60=60) Hill matrices.
 *               No large sparse solvers needed.
 *
 * V = start/stop video (ffmpeg pipe -> output.mp4)
 *
 * Build Linux:
 *   gcc -O2 -march=native -std=c17 -Wall -Wextra -DNTHREADS=4 \
 *       $(sdl2-config --cflags) schrodinger.c -o schrodinger \
 *       $(sdl2-config --libs) -llapacke -llapack -lblas -lm -lpthread
 *
 * Build macOS (SDL2 + OpenBLAS from Homebrew):
 *   brew install sdl2 openblas
 *   gcc -O2 -std=c17 -Wall -Wextra -DNTHREADS=10 \
 *       -I$(brew --prefix sdl2)/include/SDL2 \
 *       -I$(brew --prefix openblas)/include \
 *       schrodinger.c -o schrodinger \
 *       -L$(brew --prefix sdl2)/lib -lSDL2 \
 *       -L$(brew --prefix openblas)/lib -lopenblas \
 *       -lm -lpthread
 */

#define _GNU_SOURCE
#include <SDL2/SDL.h>
#include <lapacke.h>
#include <gsl/gsl_sf_mathieu.h>
#include <complex.h>
#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <stdbool.h>
#include <pthread.h>

#ifndef M_PI
#define M_PI 3.14159265358979323846
#endif

#ifndef NTHREADS
#define NTHREADS 4
#endif

/* == grid ======================================================== */
#define N     1024
#define L     1.0
#define DX    (2.0*L/(N-1))
#define DY    DX
#define DT    0.000015
#define COEFF 1.0   /* hbar^2/2m, hbar=1, m=1/2 */

/* == window ====================================================== */
#define SIM_W  768
#define SIM_H  768
#define CTRL_W 300
#define WIN_W  (SIM_W + CTRL_W)
#define WIN_H  SIM_H

/* == control panel layout ======================================== */
#define CTRL_TITLE_Y    8
#define CTRL_STATUS_Y   20
#define CTRL_PROGRESS_Y 32
#define CTRL_SLIDER_Y0  48
#define CTRL_FOOTER_Y   (WIN_H - 58)

/* == Mathieu solver parameters =================================== */
/* GSL handles angular/radial functions directly = no Hill matrix needed. */
#define NEIG   1200        /* max eigenstates to store                  */
#define Q_MAX  800.0       /* q scan upper limit (q=(kf/2)^2)           */
#define Q_MIN  0.5         /* skip spurious near-zero GSL artefacts      */
#define NSCAN  15000       /* q scan steps                               */
#define N_MAX  24          /* max angular order n                        */

/* == parameters ================================================== */
static double p_a   = 0.75;
static double p_b   = 0.50;
static double p_lam = 0.12;
static double p_sig = 0.15;
static double p_x0  = -0.35;
static double p_y0  =  0.10;
static double p_ang =  20.0;
static double p_spd =  1.0;

typedef enum { MODE_NUMERICAL, MODE_ANALYTIC } SolverMode;
static SolverMode solver_mode = MODE_NUMERICAL;

/* == wavefunction grids =========================================== */
static double complex psi_buf[N][N];  /* primary wavefunction buffer   */
static double complex psi_cmp_buf[N][N]; /* numerical comparison buf   */
static double complex (*g_psi)[N] = psi_buf; /* active psi pointer     */
static double complex tmp[N][N];
static double complex col_t[N][N];
static bool           mask[N][N];
static Uint32         pixels[SIM_H][SIM_W];

/* == comparison mode state ======================================== */
static bool   comparing     = false;
static double cmp_l2        = -1.0;
static double cmp_linf      = -1.0;
static double sim_time_cmp  = 0.0;   /* numerical time tracker      */

static bool   running    = true;
static bool   paused     = false;
static bool   reinit_req = true;
static int    step_count = 0;
static double sim_time   = 0.0;

/* == Mathieu eigenstates ========================================== */
/* Stored as 1D radial + angular lookup tables (not full N*N grids). */
#define RTAB 2048   /* radial table points:  mu in [0, mu0]  */
#define ATAB 8192   /* angular table points: nu in [0, 2*pi] */

typedef struct {
    double  rtab[RTAB];   /* Mc_n or Ms_n sampled on [0, mu0]  */
    double  atab[ATAB];   /* ce_n or se_n sampled on [0, 2*pi] */
    double  E;            /* energy eigenvalue                  */
    double  mu0;          /* boundary (stored per state)        */
    double complex c;     /* expansion coefficient <phi|psi0>   */
    int     n, parity;    /* angular order, 0=even/1=odd        */
} MeigState;

static MeigState *meig[NEIG];   /* heap-allocated per state      */
static int      meig_n      = 0;
static bool     meig_ready  = false;
static double   meig_proj_norm = 0.0;  /* sum |c_s|^2; ideally ~1.0 */
static volatile bool meig_busy  = false;
static volatile bool meig_abort = false;
static pthread_t meig_thread;

/* == Precomputed elliptic coordinate grid ======================== */
/* Allocated once in build_mask(); xi=-1 flags outside-ellipse.   */
static float *xi_grid  = NULL;   /* xi(x,y)  for each grid point  */
static float *eta_grid = NULL;   /* eta(x,y) for each grid point  */


/* == video ======================================================== */
static bool   recording = false;
static FILE  *ffpipe    = NULL;
static Uint8 *rgb_buf   = NULL;

/* ==================================================================
 * Thomas algorithm = complex tridiagonal O(n)
 * ================================================================== */
static void thomas(const double complex *restrict a,
                   const double complex *restrict b,
                   const double complex *restrict c,
                         double complex *restrict d, int n)
{
    double complex cp[n], dp[n];
    cp[0]=c[0]/b[0]; dp[0]=d[0]/b[0];
    for (int i=1;i<n;i++){
        double complex m=b[i]-a[i]*cp[i-1];
        cp[i]=c[i]/m; dp[i]=(d[i]-a[i]*dp[i-1])/m;
    }
    d[n-1]=dp[n-1];
    for (int i=n-2;i>=0;i--) d[i]=dp[i]-cp[i]*d[i+1];
}

/* ==================================================================
 * Thread pool = ADI workers
 * ================================================================== */
typedef enum { PHASE_HS1, PHASE_HS2, PHASE_NORM, PHASE_EXIT } Phase;

static pthread_barrier_t bar_start, bar_done;
static volatile Phase    cur_phase;
static volatile double complex g_rx, g_ry;
static double            partial_norm[NTHREADS];
static volatile double   g_inv_norm;

static void do_hs1(int r0, int r1)
{
    double complex rx=g_rx, ry=g_ry;
    double complex a[N],b[N],c[N],d[N];
    for (int j=r0;j<r1;j++){
        for (int i=0;i<N;i++){
            if(!mask[j][i]){a[i]=0;b[i]=1;c[i]=0;d[i]=0;continue;}
            double complex yn=g_psi[j][i];
            double complex ym=(j>0   &&mask[j-1][i])?g_psi[j-1][i]:0.0;
            double complex yp=(j<N-1 &&mask[j+1][i])?g_psi[j+1][i]:0.0;
            d[i]=yn+0.5*ry*(ym-2.0*yn+yp);
            b[i]=1.0+rx;
            a[i]=(i>0   &&mask[j][i-1])?-0.5*rx:0.0;
            c[i]=(i<N-1 &&mask[j][i+1])?-0.5*rx:0.0;
        }
        thomas(a,b,c,d,N);
        for (int i=0;i<N;i++){
            double complex v=mask[j][i]?d[i]:0.0;
            tmp[j][i]=v; col_t[i][j]=v;
        }
    }
}

static void do_hs2(int c0, int c1)
{
    double complex rx=g_rx, ry=g_ry;
    double complex a[N],b[N],cc[N],d[N];
    for (int i=c0;i<c1;i++){
        for (int j=0;j<N;j++){
            if(!mask[j][i]){a[j]=0;b[j]=1;cc[j]=0;d[j]=0;continue;}
            double complex xn=col_t[i][j];
            double complex xm=(i>0   &&mask[j][i-1])?col_t[i-1][j]:0.0;
            double complex xp=(i<N-1 &&mask[j][i+1])?col_t[i+1][j]:0.0;
            d[j]=xn+0.5*rx*(xm-2.0*xn+xp);
            b[j]=1.0+ry;
            a[j] =(j>0   &&mask[j-1][i])?-0.5*ry:0.0;
            cc[j]=(j<N-1 &&mask[j+1][i])?-0.5*ry:0.0;
        }
        thomas(a,b,cc,d,N);
        for (int j=0;j<N;j++) g_psi[j][i]=mask[j][i]?d[j]:0.0;
    }
}

static double do_norm_partial(int r0, int r1)
{
    double s=0;
    for (int j=r0;j<r1;j++) for (int i=0;i<N;i++)
        s+=creal(g_psi[j][i]*conj(g_psi[j][i]));
    return s;
}

static void do_norm_apply(int r0, int r1)
{
    double inv=g_inv_norm;
    for (int j=r0;j<r1;j++) for (int i=0;i<N;i++) g_psi[j][i]*=inv;
}

typedef struct { int tid; } WorkerArg;
static WorkerArg wargs[NTHREADS];

static void *worker(void *arg)
{
    int tid=((WorkerArg*)arg)->tid;
    int r0=tid*N/NTHREADS, r1=(tid+1)*N/NTHREADS;
    for(;;){
        pthread_barrier_wait(&bar_start);
        Phase ph=cur_phase;
        if(ph==PHASE_EXIT) break;
        if      (ph==PHASE_HS1) do_hs1(r0,r1);
        else if (ph==PHASE_HS2) do_hs2(r0,r1);
        else if (ph==PHASE_NORM){
            partial_norm[tid]=do_norm_partial(r0,r1);
            pthread_barrier_wait(&bar_done);
            pthread_barrier_wait(&bar_start);
            do_norm_apply(r0,r1);
        }
        pthread_barrier_wait(&bar_done);
    }
    return NULL;
}

static void dispatch(Phase ph)
{
    cur_phase=ph;
    pthread_barrier_wait(&bar_start);
    pthread_barrier_wait(&bar_done);
}

static pthread_t threads[NTHREADS];

static void threads_init(void)
{
    pthread_barrier_init(&bar_start,NULL,NTHREADS+1);
    pthread_barrier_init(&bar_done, NULL,NTHREADS+1);
    for (int t=0;t<NTHREADS;t++){
        wargs[t].tid=t;
        pthread_create(&threads[t],NULL,worker,&wargs[t]);
    }
}

static void threads_stop(void)
{
    cur_phase=PHASE_EXIT;
    pthread_barrier_wait(&bar_start);
    for (int t=0;t<NTHREADS;t++) pthread_join(threads[t],NULL);
    pthread_barrier_destroy(&bar_start);
    pthread_barrier_destroy(&bar_done);
}

/* ==================================================================
 * Mask + initial wavepacket
 * ================================================================== */
/* ==================================================================
 * Mask + elliptic coordinate grid precomputation
 * xi_grid and eta_grid are floats to halve memory (2=N= floats vs doubles).
 * xi=-1 flags a point outside the ellipse.
 * ================================================================== */
static void build_mask(void)
{
    double a2=p_a*p_a, b2=p_b*p_b;
    double c_foc = sqrt(p_a*p_a - p_b*p_b);   /* focal distance */
    /* clamp: if a=b then c=0; avoid division by zero */
    if(c_foc < 1e-6) c_foc = 1e-6;

    /* (re)allocate grid arrays if needed */
    if(!xi_grid)  xi_grid  = malloc((size_t)N*N*sizeof(float));
    if(!eta_grid) eta_grid = malloc((size_t)N*N*sizeof(float));

    for(int j=0;j<N;j++){
        double y=-L+j*DY;
        for(int i=0;i<N;i++){
            double x=-L+i*DX;
            int k=j*N+i;
            if(x*x/a2+y*y/b2 > 1.0){
                mask[j][i]=false;
                xi_grid[k]=-1.0f; eta_grid[k]=0.0f;
                continue;
            }
            mask[j][i]=true;
            double r1=sqrt((x+c_foc)*(x+c_foc)+y*y);
            double r2=sqrt((x-c_foc)*(x-c_foc)+y*y);
            double ch=(r1+r2)/(2.0*c_foc);
            if(ch<1.0) ch=1.0;   /* numerical safety */
            double xi_v=acosh(ch);
            double ce=(r1-r2)/(2.0*c_foc);
            if(ce> 1.0) ce= 1.0;
            if(ce<-1.0) ce=-1.0;
            double eta_v=acos(ce);
            if(y<0.0) eta_v=2.0*M_PI-eta_v;
            xi_grid[k] =(float)xi_v;
            eta_grid[k]=(float)eta_v;
        }
    }
}

/* ==================================================================
 * Initial Gaussian wavepacket
 * ================================================================== */
static void init_psi(void)
{
    double kabs=2.0*M_PI/p_lam;
    double ang=p_ang*M_PI/180.0;
    double kx=kabs*cos(ang), ky=kabs*sin(ang);
    double s2=2.0*p_sig*p_sig, norm=0;
    for(int j=0;j<N;j++){
        double y=-L+j*DY;
        for(int i=0;i<N;i++){
            double x=-L+i*DX;
            if(!mask[j][i]){g_psi[j][i]=0;continue;}
            double r2=(x-p_x0)*(x-p_x0)+(y-p_y0)*(y-p_y0);
            g_psi[j][i]=exp(-r2/s2)*cexp(I*(kx*x+ky*y));
            norm+=creal(g_psi[j][i]*conj(g_psi[j][i]));
        }
    }
    norm=sqrt(norm*DX*DY);
    if(norm>1e-12) for(int j=0;j<N;j++) for(int i=0;i<N;i++) g_psi[j][i]/=norm;
    step_count=0; sim_time=0.0;
    /* reset comparison state */
    memcpy(psi_cmp_buf, psi_buf, sizeof psi_buf);
    sim_time_cmp=0.0; cmp_l2=-1.0; cmp_linf=-1.0;
}

/* ==================================================================
 * Numerical ADI step (unchanged)
 * ================================================================== */
/* step_adi_on: run one ADI step on an arbitrary buffer.
 * Temporarily swaps g_psi, steps, restores.
 * dt_phys: physical time step to use (allows comparison at fixed dt). */
static void step_adi_on(double complex buf[N][N], double dt_phys,
                         double *time_acc)
{
    double complex (*saved)[N] = g_psi;
    g_psi = buf;
    g_rx = I*COEFF*dt_phys/(DX*DX);
    g_ry = I*COEFF*dt_phys/(DY*DY);
    dispatch(PHASE_HS1);
    dispatch(PHASE_HS2);
    cur_phase=PHASE_NORM;
    pthread_barrier_wait(&bar_start);
    pthread_barrier_wait(&bar_done);
    double norm=0;
    for(int t=0;t<NTHREADS;t++) norm+=partial_norm[t];
    norm=sqrt(norm*DX*DY);
    g_inv_norm=(norm>1e-12)?1.0/norm:1.0;
    pthread_barrier_wait(&bar_start);
    pthread_barrier_wait(&bar_done);
    g_psi = saved;
    if (time_acc) *time_acc += dt_phys;
}

static void step_adi(void)
{
    double dt=DT*p_spd;
    step_adi_on(psi_buf, dt, &sim_time);
    step_count++;
}

/* ==================================================================
 * Mathieu analytic eigensolver
 *
 * Physical basis:
 *  -=== = k== in elliptic coordinates separates into:
 *   Angular: ='' + (a - 2q cos2=) = = 0          [Mathieu]
 *   Radial:  R''  - (a - 2q cosh2=) R = 0         [modified Mathieu]
 *   with q = (k=c/2)=,  shared constant a.
 *
 * Four independent angular families (2 even, 2 odd in =):
 *   fam 0: ce-even  cos(2m=)      Hill size MH+1
 *   fam 1: ce-odd   cos((2m+1)=)  Hill size MH
 *   fam 2: se-even  sin((2m+2)=)  Hill size MH
 *   fam 3: se-odd   sin((2m+1)=)  Hill size MH
 *
 * Two radial parities per family:
 *   rpar 0: even (Mc), R(0)=1, R'(0)=0
 *   rpar 1: odd  (Ms), R(0)=0, R'(0)=1
 *
 * Algorithm:
 *  For each (fam, r, rpar): scan k, at each k solve ~30=30 Hill
 *  matrix (LAPACKE_dsyev), shoot Numerov radial ODE, detect sign
 *  change in R(==), bisect to find k_{r,s}, then synthesise the
 *  2D eigenfunction =(x,y) on the Cartesian grid.
 *
 * Speed optimisations:
 *  - Eigenvalue-only Hill solve during scan (no eigenvectors)
 *  - Eigenvectors computed only at the found zero (one extra solve)
 *  - (=,=) grid precomputed in build_mask = no transcendentals per state
 *  - Angular Fourier sum via Chebyshev recurrence = O(MH) multiplies,
 *    no repeated cos() calls
 * ================================================================== */

/* ==================================================================
 * Mathieu analytic eigensolver = using GSL
 *
 * GSL provides:
 *   gsl_sf_mathieu_ce(n, q, nu)       = angular ce_n(nu; q)
 *   gsl_sf_mathieu_se(n, q, nu)       = angular se_n(nu; q)
 *   gsl_sf_mathieu_Mc(1, n, q, mu)    = radial Mc_n^(1)(mu; q)  [first kind, interior]
 *   gsl_sf_mathieu_Ms(1, n, q, mu)    = radial Ms_n^(1)(mu; q)
 *
 * Eigenstates of -nabla^2 in the ellipse with Dirichlet BC:
 *   phi_{n,r}(x,y) = Mc_n^(1)(mu; q_{n,r}) * ce_n(nu; q_{n,r})   [even]
 *   phi_{n,r}(x,y) = Ms_n^(1)(mu; q_{n,r}) * se_n(nu; q_{n,r})   [odd]
 * where q_{n,r} is the r-th zero of Mc_n^(1)(mu0; q) or Ms_n^(1)(mu0; q).
 *
 * Energy: E = 4q / f^2   (with hbar=1, m=1/2, f=focal distance)
 *
 * Algorithm:
 *   Scan q from Q_MIN to Q_MAX, detect sign changes in
 *   Mc(1,n,q,mu0) and Ms(1,n,q,mu0), bisect each sign change.
 *   For each found q_{n,r}: synthesise phi on the Cartesian grid
 *   using the precomputed (xi,eta) = (mu,nu) grid.
 *   Normalise phi on the Cartesian grid.
 * ================================================================== */

/* == Background thread: find zeros of Mc/Ms, synthesise grids ==== */
static void *mathieu_bg(void *arg)
{
    (void)arg;
    meig_n = 0;
    meig_ready = false;

    double a_ell = p_a, b_ell = p_b;
    if (b_ell >= a_ell) b_ell = a_ell - 0.02;
    double f   = sqrt(a_ell*a_ell - b_ell*b_ell);
    if (f < 1e-6) f = 1e-6;
    double mu0 = acosh(a_ell / f);

    printf("[Mathieu/GSL] a=%.3f b=%.3f f=%.5f mu0=%.5f\n",
           a_ell, b_ell, f, mu0);
    fflush(stdout);

    /* ---- scan for eigenvalues ------------------------------------ */
    typedef struct { double q; int n, parity; } Found;
    Found *found = malloc(NEIG*4 * sizeof(Found));
    if (!found) { meig_busy=false; return NULL; }
    int nf = 0;
    double dq = (Q_MAX - Q_MIN) / NSCAN;

    for (int n = 0; n <= N_MAX && !meig_abort; n++) {
        double q  = Q_MIN;
        double fp = gsl_sf_mathieu_Mc(1, n, q, mu0);
        for (int i = 1; i <= NSCAN && nf < NEIG*4; i++) {
            if (meig_abort) goto done_scan;
            double q2 = Q_MIN + i*dq;
            double f2 = gsl_sf_mathieu_Mc(1, n, q2, mu0);
            if (fp*f2 < 0.0) {
                double qa=q, qb=q2, fa=fp;
                for (int it=0;it<64;it++) {
                    double qm=0.5*(qa+qb);
                    double fm=gsl_sf_mathieu_Mc(1,n,qm,mu0);
                    if (fa*fm<=0.0) qb=qm; else{qa=qm;fa=fm;}
                }
                found[nf++]=(Found){0.5*(qa+qb),n,0};
            }
            q=q2; fp=f2;
        }
    }
    for (int n = 1; n <= N_MAX && !meig_abort; n++) {
        double q  = Q_MIN;
        double fp = gsl_sf_mathieu_Ms(1, n, q, mu0);
        for (int i = 1; i <= NSCAN && nf < NEIG*4; i++) {
            if (meig_abort) goto done_scan;
            double q2 = Q_MIN + i*dq;
            double f2 = gsl_sf_mathieu_Ms(1, n, q2, mu0);
            if (fp*f2 < 0.0) {
                double qa=q, qb=q2, fa=fp;
                for (int it=0;it<64;it++) {
                    double qm=0.5*(qa+qb);
                    double fm=gsl_sf_mathieu_Ms(1,n,qm,mu0);
                    if (fa*fm<=0.0) qb=qm; else{qa=qm;fa=fm;}
                }
                found[nf++]=(Found){0.5*(qa+qb),n,1};
            }
            q=q2; fp=f2;
        }
    }
done_scan:
    if (meig_abort) { free(found); meig_busy=false; return NULL; }

    /* sort by q, deduplicate */
    for (int i=0;i<nf-1;i++) for(int j=i+1;j<nf;j++)
        if (found[j].q < found[i].q) { Found t=found[i];found[i]=found[j];found[j]=t; }
    Found *dedup = malloc(NEIG*4 * sizeof(Found)); int nd=0;
    if (!dedup) { free(found); meig_busy=false; return NULL; }
    for (int i=0;i<nf;i++) {
        if (i>0 && fabs(found[i].q-found[i-1].q)<1e-4) continue;
        dedup[nd++]=found[i];
    }
    int nstore = (nd < NEIG) ? nd : NEIG;
    printf("[Mathieu/GSL] %d eigenvalues, processing %d states\n", nd, nstore);
    fflush(stdout);

    /* ---- build psi0 once for projection -------------------------- */
    double kabs = 2.0*M_PI / p_lam;
    double ang  = p_ang * M_PI / 180.0;
    double kx0  = kabs*cos(ang), ky0 = kabs*sin(ang);
    double s2   = 2.0*p_sig*p_sig;
    double *psi0r = malloc((size_t)N*N*sizeof(double));
    double *psi0i = malloc((size_t)N*N*sizeof(double));
    if (psi0r && psi0i) {
        double norm=0;
        for (int j=0;j<N;j++) {
            double y=-L+j*DY;
            for (int i=0;i<N;i++) {
                double x=-L+i*DX; int k=j*N+i;
                if (!mask[j][i]) { psi0r[k]=0; psi0i[k]=0; continue; }
                double r2=(x-p_x0)*(x-p_x0)+(y-p_y0)*(y-p_y0);
                double env=exp(-r2/s2), ph=kx0*x+ky0*y;
                psi0r[k]=env*cos(ph); psi0i[k]=env*sin(ph);
                norm+=env*env;
            }
        }
        norm=sqrt(norm*DX*DY);
        if (norm<1e-14) norm=1.0;
        double inv=1.0/norm;
        for (int k=0;k<N*N;k++) { psi0r[k]*=inv; psi0i[k]*=inv; }
    }

    /* ---- per-state: build 1D tables + project -------------------- */
    double inv_mu0 = (RTAB-1) / mu0;
    double inv_2pi = (ATAB-1) / (2.0*M_PI);
    double nc = 0.0;

    for (int s=0; s<nstore && !meig_abort; s++) {
        double q  = dedup[s].q;
        int    n  = dedup[s].n;
        int    par= dedup[s].parity;

        if (!meig[s]) meig[s]=malloc(sizeof(MeigState));
        if (!meig[s]) break;
        MeigState *ms = meig[s];
        ms->n=n; ms->parity=par; ms->mu0=mu0;
        ms->E = 4.0*q/(f*f);

        /* radial table: RTAB GSL calls */
        for (int k=0;k<RTAB;k++) {
            double mu_v=mu0*k/(RTAB-1);
            ms->rtab[k] = par==0
                ? gsl_sf_mathieu_Mc(1,n,q,mu_v)
                : gsl_sf_mathieu_Ms(1,n,q,mu_v);
        }
        /* angular table: ATAB GSL calls */
        for (int k=0;k<ATAB;k++) {
            double nu_v=2.0*M_PI*k/(ATAB-1);
            ms->atab[k] = par==0
                ? gsl_sf_mathieu_ce(n,q,nu_v)
                : gsl_sf_mathieu_se(n,q,nu_v);
        }

        /* single pass: normalise phi AND project onto psi0 */
        double norm=0, cr=0, ci=0;
        for (int j=0;j<N;j++) for(int i=0;i<N;i++) {
            int idx=j*N+i;
            if (xi_grid[idx]<0.0f) continue;
            double tr=(double)xi_grid[idx]*inv_mu0;
            int kr=(int)tr; if(kr>=RTAB-1)kr=RTAB-2;
            double R=ms->rtab[kr]+(tr-kr)*(ms->rtab[kr+1]-ms->rtab[kr]);
            double ta=(double)eta_grid[idx]*inv_2pi;
            int ka=(int)ta; if(ka>=ATAB-1)ka=ATAB-2;
            double Th=ms->atab[ka]+(ta-ka)*(ms->atab[ka+1]-ms->atab[ka]);
            double phi_v=R*Th;
            norm+=phi_v*phi_v;
            if (psi0r) { cr+=phi_v*psi0r[idx]; ci+=phi_v*psi0i[idx]; }
        }
        /* normalise: divide only rtab by norm so that phi_new = (R/norm)*Th = phi/norm.
         * Do NOT divide atab -- dividing both would give phi/norm^2.             */
        norm=sqrt(norm*DX*DY);
        if (norm>1e-14) {
            double inv=1.0/norm;
            for (int k=0;k<RTAB;k++) ms->rtab[k]*=inv;
            /* atab unchanged: Th stays as-is */
            cr*=inv; ci*=inv;
        }
        ms->c=(cr+I*ci)*DX*DY;
        nc+=creal(ms->c*conj(ms->c));
        meig_n=s+1;

        if (s%10==0||s==nstore-1) {
            printf("[Mathieu/GSL] state %3d/%d  n=%d %s  q=%.4f  E=%.5f  |c|=%.4f\n",
                   s+1,nstore,n,par?"odd":"even",q,ms->E,cabs(ms->c));
            fflush(stdout);
        }
    }

    free(psi0r); free(psi0i);
    free(found); free(dedup);
    printf("[Mathieu/GSL] norm captured: %.4f in %d states\n", nc, meig_n);
    fflush(stdout);
    meig_proj_norm = nc;

    if (!meig_abort) meig_ready=true;
    printf("[Mathieu/GSL] done.\n"); fflush(stdout);
    meig_busy=false;
    return NULL;
}

static void mathieu_start_bg(void)
{
    if (meig_busy) return;
    meig_abort = false;
    meig_busy  = true;
    meig_ready = false;
    meig_n     = 0;
    meig_proj_norm = 0.0;
    pthread_create(&meig_thread, NULL, mathieu_bg, NULL);
}

static void mathieu_stop_bg(void)
{
    if (!meig_busy) return;
    meig_abort = true;
    pthread_join(meig_thread, NULL);
    meig_busy = false;
}

/* projection is now done inline during synthesis — no separate step */
static void mathieu_project(void)
{
    /* coefficients already computed in mathieu_bg; just reset time */
    sim_time = 0.0;
}

/* == Exact time evolution using 1D tables ======================== */
static void mathieu_evolve(double dt_phys)
{
    if (!meig_ready || meig_n == 0 || !meig[0]) return;
    sim_time += dt_phys;
    memset(g_psi, 0, (size_t)N*N*sizeof(double complex));

    double inv_mu0 = (RTAB - 1) / meig[0]->mu0;
    double inv_2pi = (ATAB - 1) / (2.0 * M_PI);

    for (int s = 0; s < meig_n; s++) {
        MeigState *ms = meig[s];
        if (cabs(ms->c) < 1e-12) continue;
        double complex coeff = ms->c * cexp(-I * ms->E * sim_time);

        for (int j = 0; j < N; j++) {
            for (int i = 0; i < N; i++) {
                int idx = j*N + i;
                if (xi_grid[idx] < 0.0f) continue;

                double tr = (double)xi_grid[idx] * inv_mu0;
                int    kr = (int)tr; if (kr >= RTAB-1) kr = RTAB-2;
                double R  = ms->rtab[kr]+(tr-kr)*(ms->rtab[kr+1]-ms->rtab[kr]);

                double ta = (double)eta_grid[idx] * inv_2pi;
                int    ka = (int)ta; if (ka >= ATAB-1) ka = ATAB-2;
                double Th = ms->atab[ka]+(ta-ka)*(ms->atab[ka+1]-ms->atab[ka]);

                g_psi[j][i] += coeff * (R * Th);
            }
        }
    }
    step_count++;
}

/*
 * Video
 * ================================================================== */
static bool video_start(void)
{
    if (recording) return true;
    char cmd[512];
    snprintf(cmd,sizeof cmd,
        "ffmpeg -y -f rawvideo -pixel_format rgb24"
        " -video_size %dx%d -framerate 30 -i pipe:0"
        " -vcodec libx264 -preset fast -crf 18 -pix_fmt yuv420p"
        " output.mp4 2>/dev/null", SIM_W, SIM_H);
    ffpipe=popen(cmd,"w");
    if (!ffpipe){fprintf(stderr,"[Video] popen failed\n");return false;}
    if (!rgb_buf) rgb_buf=malloc(SIM_W*SIM_H*3);
    if (!rgb_buf){pclose(ffpipe);ffpipe=NULL;return false;}
    recording=true;
    printf("[Video] recording -> output.mp4\n"); fflush(stdout);
    return true;
}
static void video_stop(void)
{
    if (!recording) return;
    pclose(ffpipe); ffpipe=NULL; recording=false;
    printf("[Video] saved output.mp4\n"); fflush(stdout);
}
static void video_frame(void)
{
    if (!recording||!ffpipe||!rgb_buf) return;
    int k=0;
    for(int py=0;py<SIM_H;py++) for(int px=0;px<SIM_W;px++){
        Uint32 p=pixels[py][px];
        rgb_buf[k++]=(p>>16)&0xFF;
        rgb_buf[k++]=(p>>8 )&0xFF;
        rgb_buf[k++]=(p    )&0xFF;
    }
    fwrite(rgb_buf,1,SIM_W*SIM_H*3,ffpipe);
}

/* ==================================================================
 * 5x7 bitmap font
 * ================================================================== */
static const Uint8 font5x7[][5]={
/*32*/{0x00,0x00,0x00,0x00,0x00},/*33*/{0x00,0x00,0x5F,0x00,0x00},
/*34*/{0x00,0x07,0x00,0x07,0x00},/*35*/{0x14,0x7F,0x14,0x7F,0x14},
/*36*/{0x24,0x2A,0x7F,0x2A,0x12},/*37*/{0x23,0x13,0x08,0x64,0x62},
/*38*/{0x36,0x49,0x55,0x22,0x50},/*39*/{0x00,0x05,0x03,0x00,0x00},
/*40*/{0x00,0x1C,0x22,0x41,0x00},/*41*/{0x00,0x41,0x22,0x1C,0x00},
/*42*/{0x14,0x08,0x3E,0x08,0x14},/*43*/{0x08,0x08,0x3E,0x08,0x08},
/*44*/{0x00,0x50,0x30,0x00,0x00},/*45*/{0x08,0x08,0x08,0x08,0x08},
/*46*/{0x00,0x60,0x60,0x00,0x00},/*47*/{0x20,0x10,0x08,0x04,0x02},
/*48*/{0x3E,0x51,0x49,0x45,0x3E},/*49*/{0x00,0x42,0x7F,0x40,0x00},
/*50*/{0x42,0x61,0x51,0x49,0x46},/*51*/{0x21,0x41,0x45,0x4B,0x31},
/*52*/{0x18,0x14,0x12,0x7F,0x10},/*53*/{0x27,0x45,0x45,0x45,0x39},
/*54*/{0x3C,0x4A,0x49,0x49,0x30},/*55*/{0x01,0x71,0x09,0x05,0x03},
/*56*/{0x36,0x49,0x49,0x49,0x36},/*57*/{0x06,0x49,0x49,0x29,0x1E},
/*58*/{0x00,0x36,0x36,0x00,0x00},/*59*/{0x00,0x56,0x36,0x00,0x00},
/*60*/{0x08,0x14,0x22,0x41,0x00},/*61*/{0x14,0x14,0x14,0x14,0x14},
/*62*/{0x00,0x41,0x22,0x14,0x08},/*63*/{0x02,0x01,0x51,0x09,0x06},
/*64*/{0x32,0x49,0x79,0x41,0x3E},/*65*/{0x7E,0x11,0x11,0x11,0x7E},
/*66*/{0x7F,0x49,0x49,0x49,0x36},/*67*/{0x3E,0x41,0x41,0x41,0x22},
/*68*/{0x7F,0x41,0x41,0x22,0x1C},/*69*/{0x7F,0x49,0x49,0x49,0x41},
/*70*/{0x7F,0x09,0x09,0x09,0x01},/*71*/{0x3E,0x41,0x49,0x49,0x7A},
/*72*/{0x7F,0x08,0x08,0x08,0x7F},/*73*/{0x00,0x41,0x7F,0x41,0x00},
/*74*/{0x20,0x40,0x41,0x3F,0x01},/*75*/{0x7F,0x08,0x14,0x22,0x41},
/*76*/{0x7F,0x40,0x40,0x40,0x40},/*77*/{0x7F,0x02,0x0C,0x02,0x7F},
/*78*/{0x7F,0x04,0x08,0x10,0x7F},/*79*/{0x3E,0x41,0x41,0x41,0x3E},
/*80*/{0x7F,0x09,0x09,0x09,0x06},/*81*/{0x3E,0x41,0x51,0x21,0x5E},
/*82*/{0x7F,0x09,0x19,0x29,0x46},/*83*/{0x46,0x49,0x49,0x49,0x31},
/*84*/{0x01,0x01,0x7F,0x01,0x01},/*85*/{0x3F,0x40,0x40,0x40,0x3F},
/*86*/{0x1F,0x20,0x40,0x20,0x1F},/*87*/{0x3F,0x40,0x38,0x40,0x3F},
/*88*/{0x63,0x14,0x08,0x14,0x63},/*89*/{0x07,0x08,0x70,0x08,0x07},
/*90*/{0x61,0x51,0x49,0x45,0x43},/*91*/{0x00,0x7F,0x41,0x41,0x00},
/*92*/{0x02,0x04,0x08,0x10,0x20},/*93*/{0x00,0x41,0x41,0x7F,0x00},
/*94*/{0x04,0x02,0x01,0x02,0x04},/*95*/{0x40,0x40,0x40,0x40,0x40},
/*96*/{0x00,0x01,0x02,0x04,0x00},/*97*/{0x20,0x54,0x54,0x54,0x78},
/*98*/{0x7F,0x48,0x44,0x44,0x38},/*99*/{0x38,0x44,0x44,0x44,0x20},
/*100*/{0x38,0x44,0x44,0x48,0x7F},/*101*/{0x38,0x54,0x54,0x54,0x18},
/*102*/{0x08,0x7E,0x09,0x01,0x02},/*103*/{0x0C,0x52,0x52,0x52,0x3E},
/*104*/{0x7F,0x08,0x04,0x04,0x78},/*105*/{0x00,0x44,0x7D,0x40,0x00},
/*106*/{0x20,0x40,0x44,0x3D,0x00},/*107*/{0x7F,0x10,0x28,0x44,0x00},
/*108*/{0x00,0x41,0x7F,0x40,0x00},/*109*/{0x7C,0x04,0x18,0x04,0x78},
/*110*/{0x7C,0x08,0x04,0x04,0x78},/*111*/{0x38,0x44,0x44,0x44,0x38},
/*112*/{0x7C,0x14,0x14,0x14,0x08},/*113*/{0x08,0x14,0x14,0x18,0x7C},
/*114*/{0x7C,0x08,0x04,0x04,0x08},/*115*/{0x48,0x54,0x54,0x54,0x20},
/*116*/{0x04,0x3F,0x44,0x40,0x20},/*117*/{0x3C,0x40,0x40,0x40,0x3C},
/*118*/{0x1C,0x20,0x40,0x20,0x1C},/*119*/{0x3C,0x40,0x30,0x40,0x3C},
/*120*/{0x44,0x28,0x10,0x28,0x44},/*121*/{0x0C,0x50,0x50,0x50,0x3C},
/*122*/{0x44,0x64,0x54,0x4C,0x44},/*123*/{0x00,0x08,0x36,0x41,0x00},
/*124*/{0x00,0x00,0x7F,0x00,0x00},/*125*/{0x00,0x41,0x36,0x08,0x00},
/*126*/{0x10,0x08,0x08,0x10,0x08},
};

static void draw_char(SDL_Renderer *ren,int px,int py,
                      char ch,Uint8 r,Uint8 g,Uint8 b)
{
    int idx=(unsigned char)ch-32;
    if(idx<0||idx>=(int)(sizeof font5x7/sizeof font5x7[0])) return;
    SDL_SetRenderDrawColor(ren,r,g,b,255);
    for(int col=0;col<5;col++){
        Uint8 bits=font5x7[idx][col];
        for(int row=0;row<7;row++)
            if(bits&(1<<row)) SDL_RenderDrawPoint(ren,px+col,py+row);
    }
}

static void draw_text(SDL_Renderer *ren,int x,int y,const char *s,
                      Uint8 r,Uint8 g,Uint8 b,int scale)
{
    int cx=x;
    while(*s){
        if(scale==1){draw_char(ren,cx,y,*s,r,g,b);cx+=6;}
        else{
            int idx=(unsigned char)*s-32;
            if(idx>=0&&idx<(int)(sizeof font5x7/sizeof font5x7[0])){
                SDL_SetRenderDrawColor(ren,r,g,b,255);
                for(int c=0;c<5;c++){
                    Uint8 bits=font5x7[idx][c];
                    for(int row=0;row<7;row++)
                        if(bits&(1<<row)){
                            SDL_Rect blk={cx+c*2,y+row*2,2,2};
                            SDL_RenderFillRect(ren,&blk);
                        }
                }
            }
            cx+=12;
        }
        s++;
    }
}

/* ==================================================================
 * Sliders
 * ================================================================== */
typedef struct {
    const char *label, *unit;
    float vmin,vmax,val;
    int   x,y,w;
    bool  dragging;
    Uint8 cr,cg,cb;
} Slider;

#define NSLIDERS 8
static Slider sliders[NSLIDERS];

static void init_sliders(void)
{
    int cx=10, w=CTRL_W-20;
    /* slider area: from CTRL_SLIDER_Y0 to CTRL_FOOTER_Y, 8 sliders  */
    int avail = CTRL_FOOTER_Y - CTRL_SLIDER_Y0;
    int dy    = avail / NSLIDERS;
    int y0    = CTRL_SLIDER_Y0;
    sliders[0]=(Slider){"a semi-major"," ",0.35f,0.92f,(float)p_a,  cx,y0+0*dy,w,false,0xFF,0x80,0x00};
    sliders[1]=(Slider){"b semi-minor"," ",0.20f,0.88f,(float)p_b,  cx,y0+1*dy,w,false,0xFF,0xCC,0x00};
    sliders[2]=(Slider){"wavelength L"," ",0.04f,0.35f,(float)p_lam,cx,y0+2*dy,w,false,0x00,0xFF,0xCC};
    sliders[3]=(Slider){"sigma",       " ",0.05f,0.35f,(float)p_sig,cx,y0+3*dy,w,false,0x66,0xAA,0xFF};
    sliders[4]=(Slider){"x0",          " ",-0.60f,0.60f,(float)p_x0,cx,y0+4*dy,w,false,0xFF,0x44,0x88};
    sliders[5]=(Slider){"y0",          " ",-0.60f,0.60f,(float)p_y0,cx,y0+5*dy,w,false,0x88,0xFF,0x44};
    sliders[6]=(Slider){"angle deg",   " ",-180.f,180.f,(float)p_ang,cx,y0+6*dy,w,false,0xCC,0x88,0xFF};
    sliders[7]=(Slider){"speed x",     " ",0.25f,2.00f,(float)p_spd,cx,y0+7*dy,w,false,0xFF,0xFF,0x44};
}

static void sync_params(void)
{
    p_a=sliders[0].val; p_b=sliders[1].val;
    p_lam=sliders[2].val; p_sig=sliders[3].val;
    p_x0=sliders[4].val; p_y0=sliders[5].val;
    p_ang=sliders[6].val; p_spd=sliders[7].val;
    if(p_b>=p_a){p_b=p_a-0.02;sliders[1].val=(float)p_b;}
}

/* ==================================================================
 * Rendering
 * ================================================================== */
typedef struct{Uint8 r,g,b;}RGB;

static RGB hsv2rgb(double h,double s,double v)
{
    h=fmod(h,2*M_PI);if(h<0)h+=2*M_PI;
    double hd=h/(M_PI/3.0);int hi=(int)hd%6;
    double f=hd-(int)hd,p=v*(1-s),q=v*(1-s*f),t=v*(1-s*(1-f));
    double r,g,b;
    switch(hi){case 0:r=v;g=t;b=p;break;case 1:r=q;g=v;b=p;break;
               case 2:r=p;g=v;b=t;break;case 3:r=p;g=q;b=v;break;
               case 4:r=t;g=p;b=v;break;default:r=v;g=p;b=q;break;}
    return (RGB){(Uint8)(r*255),(Uint8)(g*255),(Uint8)(b*255)};
}


/* == Comparison: advance numerical psi_cmp to match analytic sim_time ==
 * Called each frame when comparing=true.
 * Catches up psi_cmp_buf to sim_time using fixed DT steps, then computes
 * L2 and L-inf norms of (psi_analytic - psi_numerical).               */
static void comparison_step(void)
{
    if (!comparing || !meig_ready) return;

    /* Advance numerical solution to match current analytic sim_time.
     * Each call does at most a bounded number of steps to stay real-time. */
    int max_steps_per_frame = 4;
    for (int s = 0; s < max_steps_per_frame && sim_time_cmp < sim_time; s++)
        step_adi_on(psi_cmp_buf, DT, &sim_time_cmp);

    /* Compute error between g_psi (analytic) and psi_cmp_buf (numerical) */
    double l2=0, linf=0;
    for (int j=0;j<N;j++) for (int i=0;i<N;i++){
        if (!mask[j][i]) continue;
        double complex diff = g_psi[j][i] - psi_cmp_buf[j][i];
        double d = cabs(diff);
        l2   += d*d;
        if (d > linf) linf = d;
    }
    cmp_l2   = sqrt(l2 * DX*DY);
    cmp_linf = linf;
}

static void render_sim(SDL_Texture *tex)
{
    double maxp=1e-30;
    for(int j=0;j<N;j++) for(int i=0;i<N;i++){
        double p=creal(g_psi[j][i]*conj(g_psi[j][i]));
        if(p>maxp)maxp=p;
    }
    double a2=p_a*p_a,b2=p_b*p_b;
    for(int py=0;py<SIM_H;py++){
        int j=N-1-(int)((double)py/SIM_H*N);
        if(j<0)j=0;else if(j>=N)j=N-1;
        for(int px=0;px<SIM_W;px++){
            int i=(int)((double)px/SIM_W*N);
            if(i<0)i=0;else if(i>=N)i=N-1;
            if(!mask[j][i]){pixels[py][px]=0xFF060E0D;continue;}
            double p=creal(g_psi[j][i]*conj(g_psi[j][i]));
            double ph=carg(g_psi[j][i]);
            double v=sqrt(p/maxp);if(v>1)v=1;
            v=1.0-(1.0-v)*(1.0-v)*(1.0-v);
            RGB c=hsv2rgb(ph,0.88,v);
            pixels[py][px]=0xFF000000|(c.r<<16)|(c.g<<8)|c.b;
        }
    }
    for(int py=0;py<SIM_H;py++){
        int j=N-1-(int)((double)py/SIM_H*N);
        if(j<0||j>=N)continue;
        double y=-L+j*DY;
        for(int px=0;px<SIM_W;px++){
            int i=(int)((double)px/SIM_W*N);
            if(i<0||i>=N)continue;
            double x=-L+i*DX,vv=x*x/a2+y*y/b2;
            if(vv>0.97&&vv<1.03) pixels[py][px]=0xFF22EE44;
        }
    }
    if(recording) for(int py=4;py<14;py++) for(int px=4;px<14;px++)
        pixels[py][px]=0xFFFF2020;

    /* comparison error overlay: show |psi_a - psi_n|^2 in orange/red */
    if(comparing && cmp_l2 >= 0.0){
        double maxerr=1e-30;
        for(int j=0;j<N;j++) for(int i=0;i<N;i++){
            if(!mask[j][i]) continue;
            double d=cabs(g_psi[j][i]-psi_cmp_buf[j][i]);
            if(d>maxerr) maxerr=d;
        }
        for(int py=0;py<SIM_H;py++){
            int j=N-1-(int)((double)py/SIM_H*N);
            if(j<0)j=0;else if(j>=N)j=N-1;
            for(int px=0;px<SIM_W;px++){
                int i=(int)((double)px/SIM_W*N);
                if(i<0)i=0;else if(i>=N)i=N-1;
                if(!mask[j][i]) continue;
                double d=cabs(g_psi[j][i]-psi_cmp_buf[j][i]);
                double v=d/maxerr; if(v>1)v=1;
                /* blend: original pixel * (1-v) + error colour * v */
                Uint32 orig=pixels[py][px];
                Uint8 or_=(orig>>16)&0xFF, og=(orig>>8)&0xFF, ob=orig&0xFF;
                /* error colour: black->yellow->red */
                Uint8 er=(Uint8)(255*fmin(1.0,2*v));
                Uint8 eg=(Uint8)(255*fmax(0.0,1.0-2*v));
                Uint8 eb=0;
                Uint8 r2=(Uint8)(or_*(1-v)+er*v);
                Uint8 g2=(Uint8)(og*(1-v)+eg*v);
                Uint8 b2=(Uint8)(ob*(1-v)+eb*v);
                pixels[py][px]=0xFF000000|(r2<<16)|(g2<<8)|b2;
            }
        }
    }

    SDL_UpdateTexture(tex,NULL,pixels,SIM_W*sizeof(Uint32));
}

/* == control panel ================================================== */
static void draw_controls(SDL_Renderer *ren, int ox)
{
    /* background + separator */
    SDL_SetRenderDrawColor(ren,14,14,20,255);
    SDL_Rect bg={ox,0,CTRL_W,WIN_H}; SDL_RenderFillRect(ren,&bg);
    SDL_SetRenderDrawColor(ren,50,50,60,255);
    SDL_RenderDrawLine(ren,ox,0,ox,WIN_H);

    /* == row 0: title (y=8) ======================================= */
    draw_text(ren,ox+6,CTRL_TITLE_Y,"SCHRODINGER BILLIARD",180,200,255,1);

    /* == row 1: status badges (y=20) ============================= */
    /* Mode badge = right-aligned */
    {
        bool ana=(solver_mode==MODE_ANALYTIC);
        const char *ms=ana?"ANALYTIC":"NUMERIC";
        int bw=(int)strlen(ms)*6+4;
        SDL_SetRenderDrawColor(ren,ana?0:0,ana?20:10,ana?50:35,220);
        SDL_Rect mb={ox+CTRL_W-bw-4,CTRL_STATUS_Y-2,bw+2,13};
        SDL_RenderFillRect(ren,&mb);
        draw_text(ren,ox+CTRL_W-bw-2,CTRL_STATUS_Y,ms,
                  ana?0x44:0x88,0xCC,0xFF,1);
    }
    /* Paused badge */
    if(paused){
        SDL_SetRenderDrawColor(ren,140,20,20,220);
        SDL_Rect pr={ox+6,CTRL_STATUS_Y-2,46,13}; SDL_RenderFillRect(ren,&pr);
        draw_text(ren,ox+8,CTRL_STATUS_Y,"PAUSED",255,180,180,1);
    }
    /* REC badge */
    if(recording){
        SDL_SetRenderDrawColor(ren,180,0,0,255);
        SDL_Rect rr={ox+56,CTRL_STATUS_Y-2,28,13}; SDL_RenderFillRect(ren,&rr);
        draw_text(ren,ox+58,CTRL_STATUS_Y,"REC",255,80,80,1);
    }

    /* == row 2: analytic progress (y=32) ========================= */
    if(solver_mode==MODE_ANALYTIC){
        char pb[64];
        if(meig_busy){
            snprintf(pb,sizeof pb,"states %d/%d",meig_n,NEIG);
            draw_text(ren,ox+6,CTRL_PROGRESS_Y,pb,200,150,50,1);
            /* progress bar: right portion of this row */
            int bx=ox+80, bw=CTRL_W-86;
            SDL_SetRenderDrawColor(ren,40,40,55,255);
            SDL_Rect bg2={bx,CTRL_PROGRESS_Y,bw,7}; SDL_RenderFillRect(ren,&bg2);
            int fw=(meig_n<NEIG)?(meig_n*bw/NEIG):bw;
            SDL_SetRenderDrawColor(ren,200,150,50,255);
            SDL_Rect bf={bx,CTRL_PROGRESS_Y,fw,7}; SDL_RenderFillRect(ren,&bf);
        } else if(meig_ready){
            /* colour: green if norm>0.8, yellow if >0.5, red otherwise */
            Uint8 nr,ng,nb;
            if     (meig_proj_norm>0.80){nr=80; ng=200;nb=80;}
            else if(meig_proj_norm>0.50){nr=220;ng=180;nb=40;}
            else                        {nr=220;ng=60; nb=60;}
            snprintf(pb,sizeof pb,"%d states  norm=%.2f",meig_n,meig_proj_norm);
            draw_text(ren,ox+6,CTRL_PROGRESS_Y,pb,nr,ng,nb,1);
            /* warn if basis is insufficient */
            if(meig_proj_norm<0.80){
                draw_text(ren,ox+6,CTRL_PROGRESS_Y+9,
                          "low: use larger lambda",220,120,40,1);
            }
        } else {
            draw_text(ren,ox+6,CTRL_PROGRESS_Y,"press M to compute",120,120,120,1);
        }
    }

    /* == sliders (CTRL_SLIDER_Y0 .. CTRL_FOOTER_Y) ================ */
    char buf[32];
    for(int k=0;k<NSLIDERS;k++){
        Slider *s=&sliders[k];
        int tx=ox+s->x, ty=s->y;
        draw_text(ren,tx,ty,s->label,s->cr,s->cg,s->cb,1);
        snprintf(buf,sizeof buf,"%.3f",(double)s->val);
        int vx=ox+s->x+s->w-(int)strlen(buf)*6;
        draw_text(ren,vx,ty,buf,200,200,200,1);
        int ty2=ty+9;
        SDL_SetRenderDrawColor(ren,35,35,45,255);
        SDL_Rect track={tx,ty2,s->w,6}; SDL_RenderFillRect(ren,&track);
        float frac=(s->val-s->vmin)/(s->vmax-s->vmin);
        int fw=(int)(frac*s->w);
        SDL_SetRenderDrawColor(ren,s->cr,s->cg,s->cb,200);
        SDL_Rect fill={tx,ty2,fw,6}; SDL_RenderFillRect(ren,&fill);
        SDL_SetRenderDrawColor(ren,230,230,230,255);
        SDL_Rect thumb={tx+fw-3,ty2-3,6,12}; SDL_RenderFillRect(ren,&thumb);
    }

    /* == footer (CTRL_FOOTER_Y .. WIN_H) ========================== */
    /* 4 rows: keys1, keys2, step/time, phase-bar+labels             */
    /* Heights: 10px per text row, 8px bar, 8px label = 36px total   */
    /* CTRL_FOOTER_Y = WIN_H-58, so:                                 */
    /*   keys1 @ WIN_H-58, keys2 @ WIN_H-48, step @ WIN_H-38        */
    /*   phase bar @ WIN_H-26 (6px tall), labels @ WIN_H-14          */
    int fy = CTRL_FOOTER_Y;
    draw_text(ren,ox+6,fy,  "M:mode C:compare V:video R:reset",100,120,100,1);
    draw_text(ren,ox+6,fy+10,"SPACE:pause  Q:quit",     100,120,100,1);
    if(solver_mode==MODE_ANALYTIC && meig_ready && meig_n>0){
        double E_max=meig[meig_n-1]->E;
        double dt_a=(M_PI/20.0)/E_max*p_spd;
        snprintf(buf,sizeof buf,"t=%.4f dt=%.1e",sim_time,dt_a);
    } else {
        snprintf(buf,sizeof buf,"step%d t=%.4f",step_count,sim_time);
    }
    draw_text(ren,ox+6,fy+20,buf,80,140,80,1);
    /* comparison error stats */
    if(comparing){
        if(cmp_l2 >= 0.0){
            snprintf(buf,sizeof buf,"L2=%.2e Linf=%.2e",cmp_l2,cmp_linf);
            draw_text(ren,ox+6,fy+30,buf,255,160,80,1);
            snprintf(buf,sizeof buf,"tn=%.4f",sim_time_cmp);
            draw_text(ren,ox+6,fy+40,buf,180,120,60,1);
        } else {
            draw_text(ren,ox+6,fy+30,"comparing...",200,150,50,1);
        }
        /* CMP badge in top status row */
        SDL_SetRenderDrawColor(ren,180,80,0,220);
        SDL_Rect cb2={ox+CTRL_W-100,CTRL_STATUS_Y-2,36,13};
        SDL_RenderFillRect(ren,&cb2);
        draw_text(ren,ox+CTRL_W-98,CTRL_STATUS_Y,"CMP",255,180,100,1);
    }

    /* phase legend = hue bar */
    int lx=ox+6, lw=CTRL_W-12, ly=fy+32;
    for(int x=0;x<lw;x++){
        double ph=(double)x/lw*2*M_PI;
        RGB c=hsv2rgb(ph,0.9,0.85);
        SDL_SetRenderDrawColor(ren,c.r,c.g,c.b,255);
        SDL_RenderDrawLine(ren,lx+x,ly,lx+x,ly+5);
    }
    draw_text(ren,lx,ly-8,"phase",130,130,155,1);
    /* -pi / pi labels below bar = must be within WIN_H */
    if(ly+6+8 <= WIN_H){
        draw_text(ren,lx,     ly+7,"-pi",120,120,130,1);
        draw_text(ren,lx+lw-18,ly+7,"pi",120,120,130,1);
    }
}

/* ==================================================================
 * Events
 * ================================================================== */
static int  drag_idx = -1;
static bool project_pending = false;

static void handle_event(SDL_Event *e, int panel_x)
{
    if(e->type==SDL_QUIT){running=false;return;}
    if(e->type==SDL_KEYDOWN){
        switch(e->key.keysym.sym){
            case SDLK_q: case SDLK_ESCAPE: running=false; break;
            case SDLK_SPACE: paused=!paused; break;
            case SDLK_r: reinit_req=true; break;
            case SDLK_c:
                /* toggle comparison: analytic vs numerical from same psi0 */
                if(solver_mode==MODE_ANALYTIC && meig_ready){
                    comparing = !comparing;
                    if(comparing){
                        /* reset both to t=0 */
                        mathieu_project();           /* resets sim_time=0    */
                        memcpy(psi_cmp_buf,psi_buf,sizeof psi_buf);
                        sim_time_cmp=0.0;
                        cmp_l2=-1.0; cmp_linf=-1.0;
                    }
                }
                break;
            case SDLK_m:
                if(solver_mode==MODE_NUMERICAL){
                    solver_mode=MODE_ANALYTIC;
                    if(!meig_busy && !meig_ready)
                        mathieu_start_bg();
                    project_pending=true;
                } else {
                    solver_mode=MODE_NUMERICAL;
                    reinit_req=true;
                    comparing=false;
                }
                break;
            case SDLK_v:
                if(!recording) video_start(); else video_stop();
                break;
        }
        return;
    }
    int mx,my; SDL_GetMouseState(&mx,&my);
    int cx=mx-panel_x;
    if(e->type==SDL_MOUSEBUTTONDOWN&&e->button.button==SDL_BUTTON_LEFT){
        for(int k=0;k<NSLIDERS;k++){
            Slider *s=&sliders[k];
            int ty2=s->y+6;
            if(cx>=s->x&&cx<=s->x+s->w&&my>=ty2-3&&my<=ty2+13){
                drag_idx=k; s->dragging=true;
                float f=(float)(cx-s->x)/s->w;
                if(f<0)f=0;else if(f>1)f=1;
                s->val=s->vmin+f*(s->vmax-s->vmin);
                sync_params(); reinit_req=true;
            }
        }
    }
    if(e->type==SDL_MOUSEBUTTONUP){
        if(drag_idx>=0){sliders[drag_idx].dragging=false;}
        drag_idx=-1;
    }
    if(e->type==SDL_MOUSEMOTION&&drag_idx>=0){
        Slider *s=&sliders[drag_idx];
        float f=(float)(cx-s->x)/s->w;
        if(f<0)f=0;else if(f>1)f=1;
        s->val=s->vmin+f*(s->vmax-s->vmin);
        sync_params(); reinit_req=true;
    }
}

/* ==================================================================
 * main
 * ================================================================== */
int main(void)
{
    if(SDL_Init(SDL_INIT_VIDEO)){
        fprintf(stderr,"SDL_Init: %s\n",SDL_GetError()); return 1;
    }
    SDL_Window *win=SDL_CreateWindow(
        "Schrodinger Billiard  [M=mode V=video SPACE=pause R=reset Q=quit]",
        SDL_WINDOWPOS_CENTERED,SDL_WINDOWPOS_CENTERED,WIN_W,WIN_H,SDL_WINDOW_SHOWN);
    SDL_Renderer *ren=SDL_CreateRenderer(win,-1,
        SDL_RENDERER_ACCELERATED|SDL_RENDERER_PRESENTVSYNC);
    SDL_Texture *tex=SDL_CreateTexture(ren,
        SDL_PIXELFORMAT_ARGB8888,SDL_TEXTUREACCESS_STREAMING,SIM_W,SIM_H);

    memset(meig,0,sizeof meig);
    int panel_x=SIM_W;
    init_sliders();
    threads_init();

    printf("NTHREADS=%d\n",NTHREADS);
    printf("M=mode  V=video  SPACE=pause  R=reset  Q=quit\n\n");

    Uint32 t_title=SDL_GetTicks();

    while(running){
        SDL_Event ev;
        while(SDL_PollEvent(&ev)) handle_event(&ev,panel_x);

        if(reinit_req){
            sync_params(); build_mask(); init_psi();
            reinit_req=false;
            if(solver_mode==MODE_ANALYTIC){
                /* stop any running bg thread, restart with new parameters */
                mathieu_stop_bg();
                mathieu_start_bg();
                project_pending=true;
            }
        }

        /* once bg finishes, reset sim_time (coefficients already computed) */
        if(project_pending&&meig_ready&&!meig_busy){
            mathieu_project(); project_pending=false;
        }

        if(!paused){
            if(solver_mode==MODE_NUMERICAL){
                for(int s=0;s<2;s++) step_adi();
            } else if(solver_mode==MODE_ANALYTIC&&meig_ready){
                /* Cap dt so the highest-energy included state advances
                 * at most pi/20 per frame (prevents aliasing/jumping).
                 * Scale by p_spd slider [0.25 .. 2].                  */
                double E_max = meig[meig_n-1]->E;
                double dt_analytic = (M_PI / 20.0) / E_max * p_spd;
                mathieu_evolve(dt_analytic);
            }
        }

        comparison_step();
        render_sim(tex);
        SDL_SetRenderDrawColor(ren,0,0,0,255);
        SDL_RenderClear(ren);
        SDL_Rect dst={0,0,SIM_W,SIM_H};
        SDL_RenderCopy(ren,tex,NULL,&dst);
        draw_controls(ren,panel_x);
        SDL_RenderPresent(ren);
        if(recording) video_frame();

        Uint32 now=SDL_GetTicks();
        if(now-t_title>1000){
            char title[180];
            snprintf(title,sizeof title,
                "Billiard|%s|step=%d t=%.3f|a=%.2f b=%.2f|%sNT=%d",
                solver_mode==MODE_ANALYTIC?"ANALYTIC":"NUMERIC",
                step_count,sim_time,p_a,p_b,
                recording?"REC ":"",NTHREADS);
            SDL_SetWindowTitle(win,title);
            t_title=now;
        }
    }

    /* == clean shutdown =========================================== */
    if(recording) video_stop();
    /* Signal ARPACK thread to abort and wait for it */
    mathieu_stop_bg();
    for(int s=0;s<NEIG;s++) free(meig[s]);
    free(rgb_buf);
    threads_stop();
    SDL_DestroyTexture(tex);
    SDL_DestroyRenderer(ren);
    SDL_DestroyWindow(win);
    SDL_Quit();
    return 0;
}
