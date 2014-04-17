#include <stdlib.h>
#include <stdio.h>
#include <assert.h>
#include <math.h>
#include <cuda.h>
#include "common.h"

#define PARTICLE_SIZE CUTOFF
#define NUM_THREADS 256
#define MAX_PARTICLES_PER_BIN 4


__global__ void initBins(int n)
{
	//	number of particles i neach bin
	//	initialilzed to zero each step
	int* binCounters = (int*) a;

	//	indicies of each particle in a particular bin
	//	only has room for MAX_PARTIVLES_PER_BIN
	int* particlesInBin = (int*) binCounters[n];
	
	//	the size of numParticlesInBin is:
	//	NUM_PARTICLES * MAX_PARTICLES_PER_BIN
	int* unused = (int*) numParticlesInBin[n*MAX_PARTICLES_PER_BIN]; 
}

__device__ int2 calBin( float2 pos)
{
	int2 binPos;
	binPos.x = floor(pos.x/PARTICLE_SIZE);
	binPos.y = floor(pos.y/PARTICLE_SIZE);
	return binPos;
}

__global__ void updateBins( particle_t *particles, int n )
{
  int tid = threadIdx.x + blockIdx.x * blockDim.x;
  if(tid >= n) return;
	
	int2 bi = calBin(particles[tid].pos);	//	calculate bin index
	atomicAdd(&binCounters[bi.x*bi.y],1)
	if(binCounters[bi.x*bi.y] >= MAX_PARTICLES_PER_BIN)
		return;	//	probably do something else for error

	//	add our particle index to the bin	
	atomicAdd(&particlesInBin[binCounters[bi.x*bi.y]], tid);
}


__global__ void compute_forces_gpu(particle_t * particles, int n)
{
  // Get thread (particle) ID
  int tid = threadIdx.x + blockIdx.x * blockDim.x;
  if(tid >= n) return;

  particles[tid].ax = particles[tid].ay = 0;
  for(int j = 0 ; j < n ; j++)
    apply_force_gpu(particles[tid], particles[j]);

}

__global__ void move_gpu (particle_t * particles, int n, double size)
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

    // GPU particle data structure
    particle_t * d_particles;
    cudaMalloc((void **) &d_particles, n * sizeof(particle_t));
		set_size( n );
		
		//	use this as an address because it makes sense to me
		extern __shared__ int a[];
		//	extern __shared__ int binIndicies[];


}
