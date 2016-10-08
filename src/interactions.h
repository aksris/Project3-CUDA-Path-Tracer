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

__host__ __device__ glm::vec2 sampleHalton(int iter) {
	glm::vec2 uv(0);
	uv.x = RadicalInverse(iter, 2);
	uv.y = RadicalInverse(iter, 3);
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
	uv.x = VanDerCorput(4, 1);
	uv.y = Sobol2(4, 1);
	return uv;
}

__host__ __device__
glm::vec3 calculateRandomDirectionInHemispherexy(
	glm::vec3 normal, thrust::default_random_engine &rng, int iter) {
	
	/*thrust::uniform_real_distribution<float> u01(0, 1);*/

	glm::vec2 xy = sampleHalton(iter);
	xy = glm::normalize(xy);
    thrust::uniform_real_distribution<float> u01(xy.x, xy.x+1);
	thrust::uniform_real_distribution<float> u02(xy.y, xy.y+1);

	float u, v;

	u = TWO_PI * xy.x;
	v = sqrt(1 - xy.y);

	/*glm::vec3 ret;
	ret.x = v * glm::cos(u);
	ret.y = sqrtf(xy.y);
	ret.z = v * glm::sin(u);
	return ret;*/

	float up = sqrtf(xy.x); // cos(theta)
	float over = sqrtf(1 - up * up); // sin(theta)
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

//__host__ __device__ glm::vec3 distribution_sample_wh(glm::vec3 wo, thrust::default_random_engine &rng) {
//	glm::vec3 wh;
//	thrust::uniform_real_distribution<float> u01(0, 1);
//	glm::vec2 u(u01(rng), u01(rng));
//	float cosTheta = 0, phi = (2 * M_PI) * u[1];
//	if (alphax == alphay) {
//		float tanTheta2 = alphax * alphax * u[0] / (1.0f - u[0]);
//		cosTheta = 1 / std::sqrt(1 + tanTheta2);
//	}
//	else {
//		phi =
//			std::atan(alphay / alphax * std::tan(2 * M_PI * u[1] + .5f * Pi));
//		if (u[1] > .5f) phi += Pi;
//		float sinPhi = std::sin(phi), cosPhi = std::cos(phi);
//		const float alphax2 = alphax * alphax, alphay2 = alphay * alphay;
//		const float alpha2 =
//			1 / (cosPhi * cosPhi / alphax2 + sinPhi * sinPhi / alphay2);
//		float tanTheta2 = alpha2 * u[0] / (1 - u[0]);
//		cosTheta = 1 / std::sqrt(1 + tanTheta2);
//	}
//	float sinTheta =
//		std::sqrt(std::max((float)0., (float)1. - cosTheta * cosTheta));
//	wh = SphericalDirection(sinTheta, cosTheta, phi);
//	if (!SameHemisphere(wo, wh)) wh = -wh;
//	return wh;
//}

__host__ __device__ float CosTheta(glm::vec3 wi, glm::vec3 n) {
	return glm::dot(n, wi);
}

__host__ __device__ float SinTheta(glm::vec3 wi, glm::vec3 n) {
	return sqrtf(glm::max(0.f, 1.f - CosTheta(wi, n) * CosTheta(wi, n)));
}

__host__ __device__ float CosPhi(glm::vec3 wi, glm::vec3 n) {
	float sintheta = SinTheta(wi, n);
	if (sintheta == 0.f) return 1.f;
	return glm::clamp(wi.x / sintheta, -1.f, 1.f);
}

__host__ __device__ float SinPhi(glm::vec3 wi, glm::vec3 n) {
	float sintheta = SinTheta(wi, n);
	if (sintheta == 0.f) return 1.f;
	return glm::clamp(wi.y / sintheta, -1.f, 1.f);
}

__host__ __device__ float AbsCosTheta(glm::vec3 wi, glm::vec3 n) {
	return glm::abs(CosTheta(wi, n));
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
		float CosTheta = glm::dot(normal, pathSegment.ray.direction);
		//change based on entering or exiting the refractive object
		if (CosTheta < 0.f) ior = 1.f / ior;

		float R0 = powf((1 - ior) / (1 + ior), 2.f);
		//schlicks coeff
		float RTheta = R0 + (1 - R0) * powf((1.f - glm::abs(CosTheta)), 5.f);

		if (RTheta < u01(rng))
			pathSegment.ray.direction = glm::normalize(glm::refract(pathSegment.ray.direction, normal, ior));
		else
			pathSegment.ray.direction = glm::normalize(glm::reflect(pathSegment.ray.direction, normal));
		
		pathSegment.color *= m.specular.color;
	}
	else if (m.specular.exponent > 0.f) {
		//microfacet sample_f
		glm::vec3 wi = pathSegment.ray.direction;
		//glm::vec3 wh = distribution_sample_wh(pathSegment.ray.direction, rng);
		pathSegment.ray.direction = /*-pathSegment.ray.direction + 2 * glm::dot(pathSegment.ray.direction, wh) * wh*/glm::normalize(calculateRandomDirectionInHemisphere(normal, rng));
		glm::vec3 wo = pathSegment.ray.direction;
		float sinThetaI = SinTheta(wi, normal);
		float sinThetaO = SinTheta(wo, normal);
		// Compute cosine term of Oren-Nayar model
		glm::vec3 tangent = glm::normalize(glm::cross(normal, wi));
		float maxCos = 0;
		if (sinThetaI > 1e-4 && sinThetaO > 1e-4) {
			float sinPhiI = SinTheta(wi, tangent), cosPhiI = CosTheta(wi, tangent);
			float sinPhiO = SinTheta(wo, tangent), cosPhiO = CosTheta(wo, tangent);
			float dCos = cosPhiI * cosPhiO + sinPhiI * sinPhiO;
			maxCos = glm::max(0.f, dCos);
		}

		// Compute sine and tangent terms of Oren-Nayar model
		float sinAlpha, tanBeta;
		if (AbsCosTheta(wi, normal) > AbsCosTheta(wo, normal)) {
			sinAlpha = sinThetaO;
			tanBeta = sinThetaI / AbsCosTheta(wi, normal);
		}
		else {
			sinAlpha = sinThetaI;
			tanBeta = sinThetaO / AbsCosTheta(wo, normal);
		}
		float sigma = glm::radians(300.f);
		float A = 1.f - ((sigma*sigma) / (2.f *(sigma*sigma + 0.33f)));
		float B = 0.45f * (sigma*sigma) / (sigma*sigma + 0.09f);
		pathSegment.color *= (A + B * maxCos * sinAlpha * tanBeta);
	}
	else {
		if (depth <= -1)
			pathSegment.ray.direction = glm::normalize(calculateRandomDirectionInHemispherexy(normal, rng, iter));
		else
			pathSegment.ray.direction = glm::normalize(calculateRandomDirectionInHemisphere(normal, rng));
		//pathSegment.ray.direction = sample_wh(pathSegment.ray.direction
		//pathSegment.color *= glm::abs(glm::dot(normal, pathSegment.ray.direction));
	}
	pathSegment.color *= m.color * glm::abs(glm::dot(normal, pathSegment.ray.direction));
	pathSegment.ray.origin = intersect + (1e-3f * pathSegment.ray.direction);
	pathSegment.remainingBounces--;
}