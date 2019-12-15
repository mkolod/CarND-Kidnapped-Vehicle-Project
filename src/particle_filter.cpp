/**
 * particle_filter.cpp
 *
 * Created on: Dec 12, 2016
 * Author: Tiffany Huang
 */

#include "particle_filter.h"

#include <math.h>
#include <algorithm>
#include <iostream>
#include <iterator>
#include <numeric>
#include <random>
#include <string>
#include <vector>
#include <unordered_map>

#include "helper_functions.h"

using std::string;
using std::vector;
using dbl_normdist = std::normal_distribution<double>;

static std::default_random_engine gen(123);

inline double multiv_prob(double sig_x, double sig_y, double x_obs, double y_obs,
                   double mu_x, double mu_y) {
  // calculate normalization term
  double gauss_norm = 1 / (2 * M_PI * sig_x * sig_y);

  // calculate exponent
  double exponent = (pow(x_obs - mu_x, 2) / (2 * pow(sig_x, 2)))
               + (pow(y_obs - mu_y, 2) / (2 * pow(sig_y, 2)));
    
  // calculate weight using normalization terms and exponent
  double weight = gauss_norm * exp(-exponent);
    
  return weight;
}

void ParticleFilter::init(double x, double y, double theta, double std[]) {
  /**
   * TODO: Set the number of particles. Initialize all particles to 
   *   first position (based on estimates of x, y, theta and their uncertainties
   *   from GPS) and all weights to 1. 
   */

  constexpr int num_particles = 100;

  dbl_normdist x_rng(x, std[0]);
  dbl_normdist y_rng(y, std[1]);
  dbl_normdist theta_rng(theta, std[2]);

  particles.reserve(num_particles); 

  #pragma unroll
  for (int i = 0; i < num_particles; ++i) {
    particles.emplace_back(Particle { 
      i, x_rng(gen), y_rng(gen), theta_rng(gen), 1.0,
      std::vector<int>(), std::vector<double>(), std::vector<double>() });
  }

  is_initialized = true;
}

void ParticleFilter::prediction(double delta_t, double std_pos[], 
                                double velocity, double yaw_rate) {
  /**
   * TODO: Add measurements to each particle and add random Gaussian noise.
   * NOTE: When adding noise you may find std::normal_distribution 
   *   and std::default_random_engine useful.
   *  http://en.cppreference.com/w/cpp/numeric/random/normal_distribution
   *  http://www.cplusplus.com/reference/random/default_random_engine/
   */

  constexpr double tolerance = 1e-5;

  dbl_normdist x_gen(0, std_pos[0]);
  dbl_normdist y_gen(0, std_pos[1]);
  dbl_normdist theta_gen(0, std_pos[2]);

  for (auto& particle : particles) { 

    const double theta = particle.theta; 
    const double v_dt = velocity * delta_t; 

    if (fabs(yaw_rate) < tolerance) {
      particle.x += v_dt * cos(theta);
      particle.y += v_dt * sin(theta);
    } else {
      const double v_yaw = velocity / yaw_rate;
      const double yaw_dt = yaw_rate * delta_t;
      particle.x += v_yaw * (sin(theta + yaw_dt) - sin(theta));
      particle.y += v_yaw * (cos(theta) - cos(theta + yaw_dt));
      particle.theta += yaw_dt;
    }

    particle.x += x_gen(gen);
    particle.y += y_gen(gen);
    particle.theta += theta_gen(gen);
  }

}

void ParticleFilter::dataAssociation(vector<LandmarkObs> predicted, 
                                     vector<LandmarkObs>& observations) {
  /**
   * TODO: Find the predicted measurement that is closest to each 
   *   observed measurement and assign the observed measurement to this 
   *   particular landmark.
   * NOTE: this method will NOT be called by the grading code. But you will 
   *   probably find it useful to implement this method and use it as a helper 
   *   during the updateWeights phase.
   */

  for (auto& observation : observations) {
     double smallest_dist = std::numeric_limits<double>::max();
     for (const auto& prediction : predicted) {
       double curr_dist = dist(observation.x, observation.y, prediction.x, prediction.y);
       if (curr_dist < smallest_dist) {
         smallest_dist = curr_dist;
	 observation.id = prediction.id;
       }
     }
   }
}

inline double weight_prob(double x, double y, double mu_x, double mu_y, double sigma_x, double sigma_y) {
  double result = 1.0 / (2 * M_PI * sigma_x * sigma_y);
  result *= exp(-( pow(x - mu_x, 2)/(2.0 * pow(sigma_x, 2)) + pow(y - mu_y, 2)/(2.0 * pow(sigma_y, 2))  ));
  return result; 
}

void ParticleFilter::updateWeights(double sensor_range, double std_landmark[], 
                                   const vector<LandmarkObs> &observations, 
                                   const Map &map_landmarks) {
  /**
   * TODO: Update the weights of each particle using a mult-variate Gaussian 
   *   distribution. You can read more about this distribution here: 
   *   https://en.wikipedia.org/wiki/Multivariate_normal_distribution
   * NOTE: The observations are given in the VEHICLE'S coordinate system. 
   *   Your particles are located according to the MAP'S coordinate system. 
   *   You will need to transform between the two systems. Keep in mind that
   *   this transformation requires both rotation AND translation (but no scaling).
   *   The following is a good resource for the theory:
   *   https://www.willamette.edu/~gorr/classes/GeneralGraphics/Transforms/transforms2d.htm
   *   and the following is a good resource for the actual equation to implement
   *   (look at equation 3.33) http://planning.cs.uiuc.edu/node99.html
   */

  const auto landmark_list = map_landmarks.landmark_list;
  for (auto& particle : particles) {
   const double x = particle.x;
   const double y = particle.y;
   const double theta = particle.theta;

   std::vector<LandmarkObs> visible_landmarks;
   std::unordered_map<int, LandmarkObs> vis_landmark_lookup;

   for (const auto& landmark : landmark_list) {
     const double distance = dist(x, y, landmark.x_f, landmark.y_f);
     if (distance <= sensor_range) {
       const auto lm_obs = LandmarkObs {landmark.id_i, landmark.x_f, landmark.y_f};
       visible_landmarks.emplace_back(lm_obs);
       vis_landmark_lookup.insert({landmark.id_i, lm_obs});
     }
   } 

   std::vector<LandmarkObs> transformed_obs;

   for (const auto& obs : observations) {
     const double trans_x = cos(theta) * obs.x - sin(theta) * obs.y + x;
     const double trans_y = sin(theta) * obs.x + cos(theta) * obs.y + y;
     const auto lm_obs = LandmarkObs {obs.id, trans_x, trans_y };
     transformed_obs.emplace_back(lm_obs);
   }

   dataAssociation(visible_landmarks, transformed_obs);

   particle.weight = 1.0;
  
  for (const auto& tobs : transformed_obs) {
     const auto found = vis_landmark_lookup.find(tobs.id);
     if (found != vis_landmark_lookup.end()) {
       const auto lm_obs = found->second;
       const double obs_weight = weight_prob(lm_obs.x, lm_obs.y, tobs.x, tobs.y, std_landmark[0], std_landmark[1]);
       particle.weight *= obs_weight;
     }
   }
  }
}

void ParticleFilter::resample() {
  /**
   * TODO: Resample particles with replacement with probability proportional 
   *   to their weight. 
   * NOTE: You may find std::discrete_distribution helpful here.
   *   http://en.cppreference.com/w/cpp/numeric/random/discrete_distribution
   */
  double max_weight = std::numeric_limits<double>::min();
  std::vector<double> weights;
  weights.reserve(num_particles);

  for (const auto& particle : particles) {
  double weight = particle.weight;
    weights.emplace_back(weight);
    if (weight > max_weight) {
      max_weight = weight;
    }
  }

  std::uniform_int_distribution<int> unif_int_dist(0, num_particles - 1);
  int index = unif_int_dist(gen);

  std::uniform_real_distribution<double> unif_real_dist(0.0, max_weight);
  double beta = 0.0;
  std::vector<Particle> resampled;
  resampled.reserve(num_particles);

  for (int i = 0; i < num_particles; ++i) {
    beta += unif_real_dist(gen) * 2.0;
    while (beta > weights[index]) {
      beta -= weights[index];
      index = (index + 1) % num_particles;
    }
    resampled.emplace_back(particles[index]);
  }
  particles = resampled;
}

void ParticleFilter::SetAssociations(Particle& particle, 
                                     const vector<int>& associations, 
                                     const vector<double>& sense_x, 
                                     const vector<double>& sense_y) {
  // particle: the particle to which assign each listed association, 
  //   and association's (x,y) world coordinates mapping
  // associations: The landmark id that goes along with each listed association
  // sense_x: the associations x mapping already converted to world coordinates
  // sense_y: the associations y mapping already converted to world coordinates
  particle.associations= associations;
  particle.sense_x = sense_x;
  particle.sense_y = sense_y;
}

string ParticleFilter::getAssociations(Particle best) {
  vector<int> v = best.associations;
  std::stringstream ss;
  copy(v.begin(), v.end(), std::ostream_iterator<int>(ss, " "));
  string s = ss.str();
  s = s.substr(0, s.length()-1);  // get rid of the trailing space
  return s;
}

string ParticleFilter::getSenseCoord(Particle best, string coord) {
  vector<double> v;

  if (coord == "X") {
    v = best.sense_x;
  } else {
    v = best.sense_y;
  }

  std::stringstream ss;
  copy(v.begin(), v.end(), std::ostream_iterator<float>(ss, " "));
  string s = ss.str();
  s = s.substr(0, s.length()-1);  // get rid of the trailing space
  return s;
}
