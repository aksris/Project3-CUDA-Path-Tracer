#pragma once
#include <cmath>
#include "intersections.h"

// CHECKITOUT
/**
 * Computes a cosine-weighted random direction in a hemisphere.
 * Used for diffuse lighting.
 */

__host__ __device__
glm::vec3 calculateRandomDirectionInHemisphere(
glm::vec3 normal, thrust::default_random_engine &rng) {

	thrust::uniform_real_distribution<float> u01(0, 1);

	float up = sqrt(u01(rng)); // cos(theta)
	float over = sqrt(1 - up * up); // sin(theta)
	float around = u01(rng) * TWO_PI;

	// Find a direction that is not the normal based off of whether or not the
	// normal's components are all equal to sqrt(1/3) or whether or not at
	// least one component is less than sqrt(1/3). Learned this trick from
	// Peter Kutz.

	glm::vec3 directionNotNormal;
	if (abs(normal.x) < SQRT_OF_ONE_THIRD) {
		directionNotNormal = glm::vec3(1, 0, 0);
	}
	else if (abs(normal.y) < SQRT_OF_ONE_THIRD) {
		directionNotNormal = glm::vec3(0, 1, 0);
	}
	else {
		directionNotNormal = glm::vec3(0, 0, 1);
	}

	// Use not-normal direction to generate two perpendicular directions
	glm::vec3 perpendicularDirection1 =
		glm::normalize(glm::cross(normal, directionNotNormal));
	glm::vec3 perpendicularDirection2 =
		glm::normalize(glm::cross(normal, perpendicularDirection1));

	return up * normal
		+ cos(around) * over * perpendicularDirection1
		+ sin(around) * over * perpendicularDirection2;
}

__host__ __device__ float RadicalInverse(int n, int base) {
	float val = 0;
	float invBase = 1.f / base, invBi = invBase;
	while (n > 0) {
		int d_i = n % base;
		val += d_i * invBi;
		n *= invBase;
		invBi *= invBase;
	}
	return val;
}

__host__ __device__ glm::ivec2 sampleStratified(int iter, int strata) {
	glm::ivec2 uv(0);
	uv.x = iter % strata;
	uv.y = iter / strata;
	return uv;
}

__host__ __device__ glm::ivec2 sampleHalton(int iter) {
	glm::ivec2 uv(0);
	uv.x = RadicalInverse(iter, 3);
	uv.y = RadicalInverse(iter, 2);
	return uv;
}

__host__ __device__ float VanDerCorput(int n, int scramble) {
	n = (n << 16) | (n >> 16);
	n = ((n & 0x00ff00ff) << 8) | ((n & 0xff00ff00) >> 8);
	n = ((n & 0x0f0f0f0f) << 4) | ((n & 0xf0f0f0f0) >> 4);
	n = ((n & 0x33333333) << 2) | ((n & 0xcccccccc) >> 2);
	n = ((n & 0x55555555) << 1) | ((n & 0xaaaaaaaa) >> 1);
	n ^= scramble;
	return ((n >> 8) & 0xffffffff) / (float)(1 << 24);
}

__host__ __device__ float Sobol2(int n, int scramble) {
	for (int v = 1 << 31; n != 0; n >> 1, v ^= v >> 1) 
		if (n & 0x1) scramble ^= v;
	return ((scramble >> 8) & 0xffffffff) / (float)(1 << 24);
}

__host__ __device__ glm::ivec2 sample02(int iter) {
	glm::ivec2 uv(0);
	uv.x = VanDerCorput(16, 1);
	uv.y = Sobol2(16, 1);
	return uv;
}

__host__ __device__
glm::vec3 calculateRandomDirectionInHemispherexy(
        glm::vec3 normal, thrust::default_random_engine &rng, int iter) {


	glm::ivec2 xy = sampleHalton(iter);
    thrust::uniform_real_distribution<float> u01(xy.x, xy.x+1);
	thrust::uniform_real_distribution<float> u02(xy.y, xy.y+1);

	float up = sqrt(xy.x); // cos(theta)
	float over = sqrt(1 - up * up); // sin(theta)
	float around = xy.y * TWO_PI;

    // Find a direction that is not the normal based off of whether or not the
    // normal's components are all equal to sqrt(1/3) or whether or not at
    // least one component is less than sqrt(1/3). Learned this trick from
    // Peter Kutz.

    glm::vec3 directionNotNormal;
    if (abs(normal.x) < SQRT_OF_ONE_THIRD) {
        directionNotNormal = glm::vec3(1, 0, 0);
    } else if (abs(normal.y) < SQRT_OF_ONE_THIRD) {
        directionNotNormal = glm::vec3(0, 1, 0);
    } else {
        directionNotNormal = glm::vec3(0, 0, 1);
    }

    // Use not-normal direction to generate two perpendicular directions
    glm::vec3 perpendicularDirection1 =
        glm::normalize(glm::cross(normal, directionNotNormal));
    glm::vec3 perpendicularDirection2 =
        glm::normalize(glm::cross(normal, perpendicularDirection1));

    return up * normal
        + cos(around) * over * perpendicularDirection1
        + sin(around) * over * perpendicularDirection2;
}

/**
 * Scatter a ray with some probabilities according to the material properties.
 * For example, a diffuse surface scatters in a cosine-weighted hemisphere.
 * A perfect specular surface scatters in the reflected ray direction.
 * In order to apply multiple effects to one surface, probabilistically choose
 * between them.
 * 
 * The visual effect you want is to straight-up add the diffuse and specular
 * components. You can do this in a few ways. This logic also applies to
 * combining other types of materias (such as refractive).
 * 
 * - Always take an even (50/50) split between a each effect (a diffuse bounce
 *   and a specular bounce), but divide the resulting color of either branch
 *   by its probability (0.5), to counteract the chance (0.5) of the branch
 *   being taken.
 *   - This way is inefficient, but serves as a good starting point - it
 *     converges slowly, especially for pure-diffuse or pure-specular.
 * - Pick the split based on the intensity of each material color, and divide
 *   branch result by that branch's probability (whatever probability you use).
 *
 * This method applies its changes to the Ray parameter `ray` in place.
 * It also modifies the color `color` of the ray in place.
 *
 * You may need to change the parameter list for your purposes!
 */
__host__ __device__
void scatterRay(
		PathSegment & pathSegment,
        glm::vec3 intersect,
        glm::vec3 normal,
        const Material &m,
        thrust::default_random_engine &rng,
		int iter,
		int depth) {
	thrust::uniform_real_distribution<float> u01(0, 1);
    // TODO: implement this.
    // A basic implementation of pure-diffuse shading will just call the
    // calculateRandomDirectionInHemisphere defined above.
	if (m.hasReflective > 0.f) {
		pathSegment.ray.direction = glm::reflect(pathSegment.ray.direction, normal);
		pathSegment.color *= m.specular.color;
	}
	else if (m.hasRefractive > 0.f) {
		//using schlicks approximation

		float ior = m.indexOfRefraction;
		//normal vs ray dir
		float costheta = glm::dot(normal, pathSegment.ray.direction);
		//change based on entering or exiting the refractive object
		if (costheta < 0.f) ior = 1.f / ior;

		float R0 = powf((1 - ior) / (1 + ior), 2.f);
		//schlicks coeff
		float RTheta = R0 + (1 - R0) * powf((1.f - glm::abs(costheta)), 5.f);

		if (RTheta < u01(rng))
			pathSegment.ray.direction = glm::normalize(glm::refract(pathSegment.ray.direction, normal, ior));
		else
			pathSegment.ray.direction = glm::normalize(glm::reflect(pathSegment.ray.direction, normal));
		
		pathSegment.color *= m.specular.color;
	}
	else {
		pathSegment.ray.direction = glm::normalize(calculateRandomDirectionInHemisphere(normal, rng));
		//pathSegment.color *= glm::abs(glm::dot(normal, pathSegment.ray.direction));
	}
	pathSegment.color *= m.color * glm::abs(glm::dot(normal, pathSegment.ray.direction));
	pathSegment.ray.origin = intersect + (EPSILON * pathSegment.ray.direction);
	pathSegment.remainingBounces--;
}
