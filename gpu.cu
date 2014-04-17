#include <stdlib.h>
#include <stdio.h>
#include <assert.h>
#include <math.h>
#include <cuda.h>
#include "common.h"

#define BIN_SIZE cutoff
#define NUM_THREADS 256
#define MPPB 8		// MAX_PARTICLES_PER_BIN was too long

extern double size;

// __device__ functions are callable from DEVICE ONLY	
__device__ int2 calBin( float x, float y)
{
	int2 binPos;
	binPos.x = floor(x/BIN_SIZE);
	binPos.y = floor(y/BIN_SIZE);
	return binPos;
}

__device__ void applyForce(particle_t &particle, particle_t &neighbor)
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
//__global__ void initBins(int n)
//{
//	//	number of particles in each bin
//	//	initialilzed to zero inbetween each step
//	int* binCounters = (int*) a;
//
//	//	indicies of each particle in a particular bin
//	//	only has room for MAX_PARTIVLES_PER_BIN
//	particle_t *particlesInBin = (particle_t*) binCounters[n];
//	
//	//	the size of numParticlesInBin is:
//	//	NUM_PARTICLES * MPPB
//	int* unused = (int*) numParticlesInBin[n*MPPB]; 
//}

__global__ void clearBins(int n)
{
	extern __shared__ int binCounters[];
	extern __shared__ particle_t* particlesInBin[];

  int tid = threadIdx.x + blockIdx.x * blockDim.x;
  if(tid >= n) return;
	binCounters[tid] = 0;
	for(int i = 0; i < MPPB; ++i)
		particlesInBin[tid*MPPB+i] = 0;	//	not using NULL, maybe should
}


//	this is called on all BINS
__global__ void binCollide(int side, int n)
{
	extern __shared__ int binCounters[];
	extern __shared__ particle_t* particlesInBin[];


  int tid = threadIdx.x + blockIdx.x * blockDim.x;
  	
	if(tid >= n) 
		return;
	
	int pITB = binCounters[tid];

	// for all particles in this bin
	for(int j = 0; j < pITB; ++j)
	{
		// reset the acceleration 
		particlesInBin[j]->ax = particlesInBin[j]->ay = 0;

		//
		// intra-bin forces
		for(int k = 0; k < binCounters[tid]; ++k)
			applyForce( *particlesInBin[tid*MPPB + j], *particlesInBin[tid*MPPB + k]);

		//	
		// left bin
		if(tid%side != 0) // if i is not leftmost in row
			for(int k = 0; k < binCounters[tid-1]; ++k)
				applyForce( *particlesInBin[tid*MPPB + j], *particlesInBin[(tid-1)*MPPB + k] );

		//	
		// right bin
		if(tid%side != side-1) // if i is not rightmost in row
			for(int k = 0; k < binCounters[tid+1]; ++k)
				applyForce( *particlesInBin[tid*MPPB + j], *particlesInBin[(tid+1)*MPPB + k] );

		//
		// up bins
		if(tid >= side) //	all but the first row 
		{
			if(tid%side > 0) // make sure we're not leftmost in row
				for(int k = 0; k < binCounters[tid-side-1]; ++k)
					applyForce( *particlesInBin[tid*MPPB + j], *particlesInBin[(tid-side-1)*MPPB+k]);

			for(int k = 0; k < binCounters[tid-side]; ++k)
					applyForce( *particlesInBin[tid*MPPB + j], *particlesInBin[(tid-side)*MPPB+k]);
		
			if(tid%side < side-1) // make sure we're not rightmost in row
				for(int k = 0; k < binCounters[tid-side+1]; ++k)
					applyForce( *particlesInBin[tid*MPPB + j], *particlesInBin[(tid-side+1)*MPPB+k]);
		}
	
		//		
		// down bins
		if(tid <= side*side - side-1) //	not sure if +1 or -1
		{
			if(tid%side > 0) // make sure we're not leftmost in row
				for(int k = 0; k < binCounters[tid+side-1]; ++k)
					applyForce( *particlesInBin[tid*MPPB + j], *particlesInBin[(tid+side-1)*MPPB+k]);
	 
			for(int k = 0; k < binCounters[tid+side]; ++k)
				applyForce( *particlesInBin[tid*MPPB + j], *particlesInBin[(tid+side)*MPPB+k]);
		
			if(tid%side < side-1) // make sure we're not rightmost in row
				for(int k = 0; k < binCounters[tid+side+1]; ++k)
					applyForce( *particlesInBin[tid*MPPB + j], *particlesInBin[(tid+side+1)*MPPB+k]); 
		}
	}
}

__global__ void moveParticles (particle_t * particles, int n, double size)
{
  // Get thread (particle) ID
  int tid = threadIdx.x + blockIdx.x * blockDim.x;
  
	if(tid >= n) 
		return;

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

//	called for all PARTICLES
__global__ void updateBins( particle_t *particles, int side, int n )
{
	extern __shared__ int binCounters[];
	extern __shared__ particle_t* particlesInBin[];


	//	threadIdx * blockIdx is the particle, 
	//	blockDim is usually just 1 (I hope)
  int tid = threadIdx.x + blockIdx.x * blockDim.x;
  
	if(tid >= n) 
		return;
	
	int2 bi = calBin(particles[tid].x, particles[tid].y);	//	calculate bin index
	atomicAdd(&binCounters[bi.x+side*bi.y],1);
	if(binCounters[bi.x*bi.y] >= MPPB || particlesInBin[binCounters[bi.x+side*bi.y]] != 0)
		return;	//	probably do something else for this error

	//	add our particle index to the bin	
	//	NOTE: this whole thing might need to be atomic?
	particlesInBin[binCounters[bi.x+side*bi.y]] = &particles[tid];
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
		int side = set_size( n );

		// GPU particle data structure
    particle_t * d_particles;

		//	//	number of particles in each bin
		//	initialilzed to zero inbetween each step
		int h_binCounters [n];
		int * binCounters;
		double copy_time = read_timer( );
		//	indicies of each particle in a particular bin
		//	only has room for MAX_PARTIVLES_PER_BIN
		particle_t h_particlesInBin[n*MPPB] ;
		particle_t *particlesInBin;

		//	allocate device memory
    cudaMalloc( (void **) &d_particles, n * sizeof(particle_t));
		cudaMalloc( (int **) &binCounters, n*sizeof(int));
		cudaMalloc( (void **) &particlesInBin, n*MPPB*sizeof(particle_t*));

		//	only the particles need to be copied to the device
		cudaMemcpy(d_particles, particles, n * sizeof(particle_t), cudaMemcpyHostToDevice);
		cudaThreadSynchronize();

		int pblks = (n + NUM_THREADS - 1) / NUM_THREADS;
		int blks = ( + NUM_THREADS -1) / NUM_THREADS;
		
		//	for all particles
		//initBins<<< pblks, NUM_THREADS >>> (n);
		updateBins<<< pblks, NUM_THREADS >>>(d_particles, side, n);
    copy_time = read_timer( ) - copy_time;
		for( int step = 0; step < NSTEPS; step++ )
		{
			//	for all bins
			binCollide<<< blks, NUM_THREADS >>>( side, n);
			cudaThreadSynchronize();	

			//	for all particles
			moveParticles <<< pblks, NUM_THREADS >>> (d_particles, n, size);
			cudaThreadSynchronize();
			
			//	for all bins
			clearBins<<<blks, NUM_THREADS >>>(n);

			cudaThreadSynchronize();
			updateBins<<< blks, NUM_THREADS >>> (d_particles, side, n);
     	
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
    double simulation_time = read_timer( ) - simulation_time;
    
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





