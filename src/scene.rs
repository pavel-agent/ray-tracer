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

#[cfg(test)]
mod tests {
    use super::*;
    use crate::ray::{Ray, Vec3};

    #[test]
    fn build_demo_scene_returns_objects() {
        let (_, objects) = build_demo_scene(16.0 / 9.0);
        // Should have at least the 3 large spheres + ground plane
        assert!(objects.len() >= 4, "Scene should have at least 4 objects, got {}", objects.len());
    }

    #[test]
    fn build_demo_scene_camera_can_generate_rays() {
        let (camera, _) = build_demo_scene(16.0 / 9.0);
        let mut rng = rand::thread_rng();
        // Camera should produce valid rays for various UV coordinates
        let ray = camera.get_ray(0.5, 0.5, &mut rng);
        assert!(ray.direction.length() > 0.0, "Ray direction should be non-zero");
    }

    #[test]
    fn build_demo_scene_objects_have_bounding_boxes() {
        let (_, objects) = build_demo_scene(1.0);
        for obj in &objects {
            let bbox = obj.bounding_box();
            // Each bounding box should have min <= max in at least some axes
            // (planes have large bounding boxes)
            assert!(bbox.max.x >= bbox.min.x || bbox.max.y >= bbox.min.y);
        }
    }

    #[test]
    fn build_demo_scene_objects_are_hittable() {
        let (_, objects) = build_demo_scene(16.0 / 9.0);
        // A ray aimed at the ground plane (y=0, normal up) from above should hit
        let ray = Ray::new(
            Point3::new(0.0, 5.0, 0.0),
            Vec3::new(0.0, -1.0, 0.0),
        );
        let mut hit_found = false;
        for obj in &objects {
            if obj.hit(&ray, 0.001, f64::INFINITY).is_some() {
                hit_found = true;
                break;
            }
        }
        assert!(hit_found, "A downward ray should hit the ground plane");
    }

    #[test]
    fn build_demo_scene_different_aspect_ratios() {
        // Should not panic for various aspect ratios
        let (_cam1, objs1) = build_demo_scene(1.0);
        let (_cam2, objs2) = build_demo_scene(2.0);
        let (_cam3, objs3) = build_demo_scene(0.5);
        // Object count should be the same regardless of aspect ratio
        // (aspect ratio only affects the camera, not the scene)
        // Note: random generation means counts may vary between calls,
        // but all should have at least the ground plane + 3 big spheres
        assert!(objs1.len() >= 4);
        assert!(objs2.len() >= 4);
        assert!(objs3.len() >= 4);
    }

    #[test]
    fn build_demo_scene_large_spheres_present() {
        let (_, objects) = build_demo_scene(16.0 / 9.0);
        // Test that a ray aimed at each large sphere center can hit something
        let large_sphere_centers = [
            Point3::new(0.0, 1.0, 0.0),
            Point3::new(-4.0, 1.0, 0.0),
            Point3::new(4.0, 1.0, 0.0),
        ];
        for center in &large_sphere_centers {
            let ray = Ray::new(
                Point3::new(center.x, center.y, center.z + 10.0),
                Vec3::new(0.0, 0.0, -1.0),
            );
            let mut hit_found = false;
            for obj in &objects {
                if obj.hit(&ray, 0.001, f64::INFINITY).is_some() {
                    hit_found = true;
                    break;
                }
            }
            assert!(hit_found, "Should hit a large sphere at {:?}", center);
        }
    }
}
