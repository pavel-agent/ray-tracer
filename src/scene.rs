use std::sync::Arc;

use rand::Rng;

use crate::camera::Camera;
use crate::hittable::{Hittable, Plane, Sphere};
use crate::material::Material;
use crate::ray::{Color, Point3, Vec3};

pub fn build_demo_scene(aspect_ratio: f64) -> (Camera, Vec<Box<dyn Hittable>>) {
    let mut objects: Vec<Box<dyn Hittable>> = Vec::new();
    let mut rng = rand::thread_rng();

    // Ground plane
    let ground_material = Arc::new(Material::Lambertian {
        albedo: Color::new(0.5, 0.5, 0.5),
    });
    objects.push(Box::new(Plane::new(
        Point3::new(0.0, 0.0, 0.0),
        Vec3::new(0.0, 1.0, 0.0),
        ground_material,
    )));

    // Random small spheres
    for a in -5..5 {
        for b in -5..5 {
            let choose_mat: f64 = rng.gen();
            let center = Point3::new(
                a as f64 + 0.9 * rng.gen::<f64>(),
                0.2,
                b as f64 + 0.9 * rng.gen::<f64>(),
            );

            if (center - Point3::new(4.0, 0.2, 0.0)).length() < 0.9 {
                continue;
            }

            let material: Arc<Material> = if choose_mat < 0.6 {
                let albedo = Color::random(&mut rng) * Color::random(&mut rng);
                Arc::new(Material::Lambertian { albedo })
            } else if choose_mat < 0.85 {
                let albedo = Color::random_range(&mut rng, 0.5, 1.0);
                let fuzz = rng.gen_range(0.0..0.5);
                Arc::new(Material::Metal { albedo, fuzz })
            } else {
                Arc::new(Material::Dielectric { ior: 1.5 })
            };

            objects.push(Box::new(Sphere::new(center, 0.2, material)));
        }
    }

    // Three large spheres
    objects.push(Box::new(Sphere::new(
        Point3::new(0.0, 1.0, 0.0),
        1.0,
        Arc::new(Material::Dielectric { ior: 1.5 }),
    )));
    objects.push(Box::new(Sphere::new(
        Point3::new(-4.0, 1.0, 0.0),
        1.0,
        Arc::new(Material::Lambertian {
            albedo: Color::new(0.4, 0.2, 0.1),
        }),
    )));
    objects.push(Box::new(Sphere::new(
        Point3::new(4.0, 1.0, 0.0),
        1.0,
        Arc::new(Material::Metal {
            albedo: Color::new(0.7, 0.6, 0.5),
            fuzz: 0.0,
        }),
    )));

    // Camera
    let lookfrom = Point3::new(13.0, 2.0, 3.0);
    let lookat = Point3::new(0.0, 0.0, 0.0);
    let vup = Vec3::new(0.0, 1.0, 0.0);
    let focus_dist = 10.0;
    let aperture = 0.1;

    let camera = Camera::new(lookfrom, lookat, vup, 20.0, aspect_ratio, aperture, focus_dist);

    (camera, objects)
}
