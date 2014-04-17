#include <stdlib.h>
#include <stdio.h>
#include <assert.h>
#include <math.h>
#include <cuda.h>
#include "common.h"

#define PARTICLE_SIZE CUTOFF
#define NUM_THREADS 256
#define MPPB 8		// MAX_PARTICLES_PER_BIN was too long


//	this is a kernel-callable function ONLY
__device__ int2 calBin( float2 pos)
{
	int2 binPos;
	binPos.x = floor(pos.x/PARTICLE_SIZE);
	binPos.y = floor(pos.y/PARTICLE_SIZE);
	return binPos;
}

__device__ void apply_force_gpu(particle_t &particle, particle_t &neighbor)
{
  double dx = neighbor.x - particle.x;
  double dy = neighbor.y - particle.y;
  double r2 = dx * dx + dy * dy;
  if( r2 > cutoff*cutoff )
      return;
  //r2 = fmax( r2, min_r*min_r );
  r2 = (r2 > min_r*min_r) ? r2 : min_r*min_r;
  double r = sqrt( r2 );

  //
  //  very simple short-range repulsive force
  //
  double coef = ( 1 - cutoff / r ) / r2 / mass;
  particle.ax += coef * dx;
  particle.ay += coef * dy;

}


//	this is a kernel
__global__ void initBins(int n)
{
	//	number of particles in each bin
	//	initialilzed to zero each step
	int* binCounters = (int*) a;

	//	indicies of each particle in a particular bin
	//	only has room for MAX_PARTIVLES_PER_BIN
	int* particlesInBin = (int*) binCounters[n];
	
	//	the size of numParticlesInBin is:
	//	NUM_PARTICLES * MPPB
	int* unused = (int*) numParticlesInBin[n*MPPB]; 
}

//	this is called on all BINS
__global__ void binCollide(
		particle_t * particles, 
		int side, 
		int n)
{
  int tid = threadIdx.x + blockIdx.x * blockDim.x;
  if(tid >= n) return;
	
	// for all particles in this bin
	for(int j = 0; j < binCounters[tid]; ++j)
	{
		// reset the acceleration 
		particles[particlesInBin[j]] -> ax = particles[particlesInBin[j]] -> ay = 0;

		//
		// intra-bin forces
		for(int k = 0; k < binCounters[tid*MPPB]; ++k)
			apply_force( particlesInBin[tid*MPPB + j], particlesInBin[tid*MPPB + k]);

/*** LEFT OFF ***/

		//	
		// left bin
		if(tid%side != 0) // if i is not leftmost in row
			for(int k = 0; k < binCounter[i-1]; ++k)
				apply_force( [i][j], [i-1][k] );


		//	
		// right bin
		if(i%side != side-1) // if i is not rightmost in row
			for(int k = 0; k < binCounter[i+1]; ++k)
				apply_force( [i][j], [i+1][k]);

		//
		// up bins
		if(i >= side) // rows are side-1 in size
		{
			if(i%side > 0) // make sure we're not leftmost in row
				for(int k = 0; k < binCounter[i-side-1]; ++k)
					apply_force([i][j],	[i-side-1][k]);

			for(int k = 0; k < binCounter[i-side]; ++k)
				apply_force( [i][j], [i-side][k]);
		
			if(i%side < side-1) // make sure we're not rightmost in row
				for(int k = 0; k < binCounter[i-side+1]; ++k)
					apply_force( [i][j], [i-side+1][k]); 
		}
	
		//		
		// down bins
		if(i <= binCounter[ ]- side -1)
		{
			if(i%side > 0) // make sure we're not leftmost in row
				for(int k = 0; k < binCounter[i+side-1]; ++k)
					apply_force( [i][j], [i+side-1][k];
	 
			for(int k = 0; k < binCounterl[i+side]; ++k)
				apply_force( [i][j], [i+side][k];
		
			if(i%side < side-1) // make sure we're not rightmost in row
				for(int k = 0; k < binCounter[i+side+1]; ++k)
					apply_force( [i][j], [i+side+1][k]); 
		}
	}
}

__global__ void moveParticles (particle_t * particles, int n, double size)
{

  // Get thread (particle) ID
  int tid = threadIdx.x + blockIdx.x * blockDim.x;
  if(tid >= n) return;

  particle_t * p = &particles[tid];
    //
    //  slightly simplified Velocity Verlet integration
    //  conserves energy better than explicit Euler method
    //
    p->vx += p->ax * dt;
    p->vy += p->ay * dt;
    p->x  += p->vx * dt;
    p->y  += p->vy * dt;

    //
    //  bounce from walls
    //
    while( p->x < 0 || p->x > size )
    {
        p->x  = p->x < 0 ? -(p->x) : 2*size-p->x;
        p->vx = -(p->vx);
    }
    while( p->y < 0 || p->y > size )
    {
        p->y  = p->y < 0 ? -(p->y) : 2*size-p->y;
        p->vy = -(p->vy);
    }
}

__global__ void updateBins( particle_t *particles, int n )
{
	//	threadIdx * blockIdx is the particle, 
	//	blockDim is usually just 1 (I hope)
  int tid = threadIdx.x + blockIdx.x * blockDim.x;
  if(tid >= n) return;
	
	int2 bi = calBin(particles[tid].pos);	//	calculate bin index
	atomicAdd(&binCounters[bi.x*bi.y],1)
	if(binCounters[bi.x*bi.y] >= MPPB)
		return;	//	probably do something else for this error

	//	add our particle index to the bin	
	atomicAdd(&particlesInBin[binCounters[bi.x*bi.y]], tid);
}

int main( int argc, char **argv )
{    
    // This takes a few seconds to initialize the runtime
    cudaThreadSynchronize(); 

    if( find_option( argc, argv, "-h" ) >= 0 )
    {
        printf( "Options:\n" );
        printf( "-h to see this help\n" );
        printf( "-n <int> to set the number of particles\n" );
        printf( "-o <filename> to specify the output file name\n" );
	printf( "-s <filename> to specify the summary output file name\n" );
        return 0;
    }
    
    int n = read_int( argc, argv, "-n", 1000 );

    char *savename = read_string( argc, argv, "-o", NULL );
    char *sumname = read_string( argc, argv, "-s", NULL );
    
    FILE *fsave = savename ? fopen( savename, "w" ) : NULL;
    FILE *fsum = sumname ? fopen(sumname,"a") : NULL;
    particle_t *particles = (particle_t*) malloc( n * sizeof(particle_t) );
		
		//	need to get number of bins somehow?
		set_size( n );
    // GPU particle data structure
    particle_t * d_particles;
    cudaMalloc((void **) &d_particles, n * sizeof(particle_t));
		
		//	use this as an address because it makes sense to me
		extern __shared__ int a[];
		//	extern __shared__ int binIndicies[];
			
		int pblks = (n + NUM_THREADS - 1) / NUM_THREADS;
		int blks = ( + NUM_THREADS -1) / NUM_THREADS;

		initBins<<< pblks, NUM_THREADS >>> (n);
		updateBins<<< pblks, NUM_THREADS >>>(d_particles, n);
    
		for( int step = 0; step < NSTEPS; step++ )
		{
			//	for all bins
			binCollide<<< blks, NUM_THREADS >>>(d_particles, side, n);

			//	for all particles
			moveParticles <<< pblks, NUM_THREADS >>> (d_particles, n, size);

			updateBins<<< blks, NUM_THREADS >>> (d_particles, n);
     	
			//	need to reset shared memory? at some point?

			//
      //  save if necessary
      //
      if( fsave && (step%SAVEFREQ) == 0 ) 
			{
	    	// Copy the particles back to the CPU
      	cudaMemcpy(particles, d_particles, n * sizeof(particle_t), cudaMemcpyDeviceToHost);
       	save( fsave, n, particles);
			}
		}

    cudaThreadSynchronize();
    simulation_time = read_timer( ) - simulation_time;
    
    printf( "CPU-GPU copy time = %g seconds\n", copy_time);
    printf( "n = %d, simulation time = %g seconds\n", n, simulation_time );

    if (fsum)
	fprintf(fsum,"%d %lf \n",n,simulation_time);

    if (fsum)
	fclose( fsum );    
    free( particles );
    cudaFree(d_particles);
    if( fsave )
        fclose( fsave );
    
    return 0;
}





