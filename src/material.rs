use rand::Rng;

use crate::hittable::HitRecord;
use crate::ray::{Color, Ray, Vec3};

pub enum Material {
    Lambertian { albedo: Color },
    Metal { albedo: Color, fuzz: f64 },
    Dielectric { ior: f64 },
}

pub struct ScatterResult {
    pub attenuation: Color,
    pub scattered: Ray,
}

impl Material {
    pub fn scatter(&self, ray_in: &Ray, rec: &HitRecord, rng: &mut impl Rng) -> Option<ScatterResult> {
        match self {
            Material::Lambertian { albedo } => {
                let mut scatter_dir = rec.normal + Vec3::random_unit_vector(rng);
                if scatter_dir.near_zero() {
                    scatter_dir = rec.normal;
                }
                Some(ScatterResult {
                    attenuation: *albedo,
                    scattered: Ray::new(rec.point, scatter_dir),
                })
            }
            Material::Metal { albedo, fuzz } => {
                let reflected = ray_in.direction.unit().reflect(rec.normal);
                let scattered = Ray::new(
                    rec.point,
                    reflected + Vec3::random_in_unit_sphere(rng) * *fuzz,
                );
                if scattered.direction.dot(rec.normal) > 0.0 {
                    Some(ScatterResult {
                        attenuation: *albedo,
                        scattered,
                    })
                } else {
                    None
                }
            }
            Material::Dielectric { ior } => {
                let refraction_ratio = if rec.front_face { 1.0 / ior } else { *ior };
                let unit_direction = ray_in.direction.unit();
                let cos_theta = (-unit_direction).dot(rec.normal).min(1.0);
                let sin_theta = (1.0 - cos_theta * cos_theta).sqrt();

                let cannot_refract = refraction_ratio * sin_theta > 1.0;
                let direction = if cannot_refract
                    || reflectance(cos_theta, refraction_ratio) > rng.gen()
                {
                    unit_direction.reflect(rec.normal)
                } else {
                    unit_direction.refract(rec.normal, refraction_ratio)
                };

                Some(ScatterResult {
                    attenuation: Color::new(1.0, 1.0, 1.0),
                    scattered: Ray::new(rec.point, direction),
                })
            }
        }
    }
}

fn reflectance(cosine: f64, ref_idx: f64) -> f64 {
    // Schlick's approximation
    let r0 = ((1.0 - ref_idx) / (1.0 + ref_idx)).powi(2);
    r0 + (1.0 - r0) * (1.0 - cosine).powi(5)
}
