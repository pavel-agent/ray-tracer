use std::sync::Arc;

use crate::material::Material;
use crate::ray::{Aabb, Point3, Ray, Vec3};

pub struct HitRecord<'a> {
    pub point: Point3,
    pub normal: Vec3,
    pub t: f64,
    pub front_face: bool,
    pub material: &'a Material,
}

impl<'a> HitRecord<'a> {
    pub fn set_face_normal(ray: &Ray, outward_normal: Vec3) -> (Vec3, bool) {
        let front_face = ray.direction.dot(outward_normal) < 0.0;
        let normal = if front_face {
            outward_normal
        } else {
            -outward_normal
        };
        (normal, front_face)
    }
}

pub trait Hittable: Send + Sync {
    fn hit(&self, ray: &Ray, t_min: f64, t_max: f64) -> Option<HitRecord>;
    fn bounding_box(&self) -> Aabb;
}

// Sphere

pub struct Sphere {
    pub center: Point3,
    pub radius: f64,
    pub material: Arc<Material>,
}

impl Sphere {
    pub fn new(center: Point3, radius: f64, material: Arc<Material>) -> Self {
        Self {
            center,
            radius,
            material,
        }
    }
}

impl Hittable for Sphere {
    fn hit(&self, ray: &Ray, t_min: f64, t_max: f64) -> Option<HitRecord> {
        let oc = ray.origin - self.center;
        let a = ray.direction.length_squared();
        let half_b = oc.dot(ray.direction);
        let c = oc.length_squared() - self.radius * self.radius;
        let discriminant = half_b * half_b - a * c;

        if discriminant < 0.0 {
            return None;
        }

        let sqrtd = discriminant.sqrt();
        let mut root = (-half_b - sqrtd) / a;
        if root < t_min || root > t_max {
            root = (-half_b + sqrtd) / a;
            if root < t_min || root > t_max {
                return None;
            }
        }

        let point = ray.at(root);
        let outward_normal = (point - self.center) / self.radius;
        let (normal, front_face) = HitRecord::set_face_normal(ray, outward_normal);

        Some(HitRecord {
            point,
            normal,
            t: root,
            front_face,
            material: &self.material,
        })
    }

    fn bounding_box(&self) -> Aabb {
        let offset = Vec3::new(self.radius, self.radius, self.radius);
        Aabb::new(self.center - offset, self.center + offset)
    }
}

// Plane (infinite plane defined by point and normal)

pub struct Plane {
    pub point: Point3,
    pub normal: Vec3,
    pub material: Arc<Material>,
}

impl Plane {
    pub fn new(point: Point3, normal: Vec3, material: Arc<Material>) -> Self {
        Self {
            point,
            normal: normal.unit(),
            material,
        }
    }
}

impl Hittable for Plane {
    fn hit(&self, ray: &Ray, t_min: f64, t_max: f64) -> Option<HitRecord> {
        let denom = ray.direction.dot(self.normal);
        if denom.abs() < 1e-8 {
            return None;
        }

        let t = (self.point - ray.origin).dot(self.normal) / denom;
        if t < t_min || t > t_max {
            return None;
        }

        let point = ray.at(t);
        let (normal, front_face) = HitRecord::set_face_normal(ray, self.normal);

        Some(HitRecord {
            point,
            normal,
            t,
            front_face,
            material: &self.material,
        })
    }

    fn bounding_box(&self) -> Aabb {
        // Infinite planes get a very large bounding box
        let big = 1e4;
        Aabb::new(
            Point3::new(-big, -big, -big),
            Point3::new(big, big, big),
        )
    }
}

// HittableList

pub struct HittableList {
    pub objects: Vec<Box<dyn Hittable>>,
}

impl HittableList {
    pub fn new() -> Self {
        Self {
            objects: Vec::new(),
        }
    }

    pub fn add(&mut self, object: Box<dyn Hittable>) {
        self.objects.push(object);
    }
}

impl Hittable for HittableList {
    fn hit(&self, ray: &Ray, t_min: f64, t_max: f64) -> Option<HitRecord> {
        let mut closest = t_max;
        let mut result = None;

        for object in &self.objects {
            if let Some(rec) = object.hit(ray, t_min, closest) {
                closest = rec.t;
                result = Some(rec);
            }
        }

        result
    }

    fn bounding_box(&self) -> Aabb {
        let mut output_box = Aabb::empty();
        for object in &self.objects {
            output_box = Aabb::surrounding_box(output_box, object.bounding_box());
        }
        output_box
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::material::Material;

    const EPSILON: f64 = 1e-6;

    fn approx_eq(a: f64, b: f64) -> bool {
        (a - b).abs() < EPSILON
    }

    fn vec3_approx_eq(a: Vec3, b: Vec3) -> bool {
        approx_eq(a.x, b.x) && approx_eq(a.y, b.y) && approx_eq(a.z, b.z)
    }

    fn test_material() -> Arc<Material> {
        Arc::new(Material::Lambertian {
            albedo: crate::ray::Color::new(0.5, 0.5, 0.5),
        })
    }

    // ---- HitRecord::set_face_normal ----

    #[test]
    fn hit_record_front_face() {
        // Ray traveling in -z, normal pointing in +z => front face
        let ray = Ray::new(Point3::new(0.0, 0.0, -5.0), Vec3::new(0.0, 0.0, 1.0));
        let outward_normal = Vec3::new(0.0, 0.0, -1.0);
        // direction dot outward_normal = 1.0 * -1.0 = -1.0, which is < 0 => ... wait
        // direction is (0,0,1), outward_normal is (0,0,-1), dot = -1 < 0 => front_face = true
        let (normal, front_face) = HitRecord::set_face_normal(&ray, outward_normal);
        assert!(front_face);
        assert!(vec3_approx_eq(normal, outward_normal));
    }

    #[test]
    fn hit_record_back_face() {
        // Ray traveling in +z, outward normal pointing in +z => back face
        let ray = Ray::new(Point3::new(0.0, 0.0, -5.0), Vec3::new(0.0, 0.0, 1.0));
        let outward_normal = Vec3::new(0.0, 0.0, 1.0);
        let (normal, front_face) = HitRecord::set_face_normal(&ray, outward_normal);
        assert!(!front_face);
        assert!(vec3_approx_eq(normal, Vec3::new(0.0, 0.0, -1.0)));
    }

    // ---- Sphere intersection tests ----

    #[test]
    fn sphere_hit_basic() {
        let sphere = Sphere::new(Point3::new(0.0, 0.0, -5.0), 1.0, test_material());
        let ray = Ray::new(Point3::new(0.0, 0.0, 0.0), Vec3::new(0.0, 0.0, -1.0));
        let hit = sphere.hit(&ray, 0.001, f64::INFINITY);
        assert!(hit.is_some());
        let rec = hit.unwrap();
        assert!(approx_eq(rec.t, 4.0)); // Hit at z = -4 (front of sphere)
        assert!(rec.front_face);
    }

    #[test]
    fn sphere_miss() {
        let sphere = Sphere::new(Point3::new(0.0, 0.0, -5.0), 1.0, test_material());
        let ray = Ray::new(Point3::new(0.0, 0.0, 0.0), Vec3::new(0.0, 1.0, 0.0));
        let hit = sphere.hit(&ray, 0.001, f64::INFINITY);
        assert!(hit.is_none());
    }

    #[test]
    fn sphere_hit_from_inside() {
        // Ray originates inside the sphere
        let sphere = Sphere::new(Point3::new(0.0, 0.0, 0.0), 10.0, test_material());
        let ray = Ray::new(Point3::new(0.0, 0.0, 0.0), Vec3::new(1.0, 0.0, 0.0));
        let hit = sphere.hit(&ray, 0.001, f64::INFINITY);
        assert!(hit.is_some());
        let rec = hit.unwrap();
        assert!(!rec.front_face); // Hitting from inside
    }

    #[test]
    fn sphere_hit_t_range() {
        let sphere = Sphere::new(Point3::new(0.0, 0.0, -5.0), 1.0, test_material());
        let ray = Ray::new(Point3::new(0.0, 0.0, 0.0), Vec3::new(0.0, 0.0, -1.0));
        // t range that excludes the sphere (sphere is at t=4..6)
        let hit = sphere.hit(&ray, 0.001, 3.0);
        assert!(hit.is_none());
    }

    #[test]
    fn sphere_hit_normal_is_unit_length() {
        let sphere = Sphere::new(Point3::new(0.0, 0.0, -5.0), 1.0, test_material());
        let ray = Ray::new(Point3::new(0.0, 0.0, 0.0), Vec3::new(0.0, 0.0, -1.0));
        let rec = sphere.hit(&ray, 0.001, f64::INFINITY).unwrap();
        assert!(approx_eq(rec.normal.length(), 1.0));
    }

    #[test]
    fn sphere_hit_point_on_surface() {
        let center = Point3::new(0.0, 0.0, -5.0);
        let radius = 1.0;
        let sphere = Sphere::new(center, radius, test_material());
        let ray = Ray::new(Point3::new(0.0, 0.0, 0.0), Vec3::new(0.0, 0.0, -1.0));
        let rec = sphere.hit(&ray, 0.001, f64::INFINITY).unwrap();
        let dist = (rec.point - center).length();
        assert!(approx_eq(dist, radius));
    }

    #[test]
    fn sphere_bounding_box() {
        let sphere = Sphere::new(Point3::new(1.0, 2.0, 3.0), 0.5, test_material());
        let bbox = sphere.bounding_box();
        assert!(vec3_approx_eq(bbox.min, Point3::new(0.5, 1.5, 2.5)));
        assert!(vec3_approx_eq(bbox.max, Point3::new(1.5, 2.5, 3.5)));
    }

    // ---- Plane intersection tests ----

    #[test]
    fn plane_hit_basic() {
        let plane = Plane::new(
            Point3::new(0.0, 0.0, 0.0),
            Vec3::new(0.0, 1.0, 0.0),
            test_material(),
        );
        let ray = Ray::new(Point3::new(0.0, 5.0, 0.0), Vec3::new(0.0, -1.0, 0.0));
        let hit = plane.hit(&ray, 0.001, f64::INFINITY);
        assert!(hit.is_some());
        let rec = hit.unwrap();
        assert!(approx_eq(rec.t, 5.0));
        assert!(rec.front_face);
    }

    #[test]
    fn plane_miss_parallel_ray() {
        let plane = Plane::new(
            Point3::new(0.0, 0.0, 0.0),
            Vec3::new(0.0, 1.0, 0.0),
            test_material(),
        );
        let ray = Ray::new(Point3::new(0.0, 5.0, 0.0), Vec3::new(1.0, 0.0, 0.0));
        let hit = plane.hit(&ray, 0.001, f64::INFINITY);
        assert!(hit.is_none());
    }

    #[test]
    fn plane_hit_from_below() {
        let plane = Plane::new(
            Point3::new(0.0, 0.0, 0.0),
            Vec3::new(0.0, 1.0, 0.0),
            test_material(),
        );
        let ray = Ray::new(Point3::new(0.0, -5.0, 0.0), Vec3::new(0.0, 1.0, 0.0));
        let hit = plane.hit(&ray, 0.001, f64::INFINITY);
        assert!(hit.is_some());
        let rec = hit.unwrap();
        assert!(!rec.front_face); // Hitting from below
    }

    #[test]
    fn plane_bounding_box_is_large() {
        let plane = Plane::new(
            Point3::new(0.0, 0.0, 0.0),
            Vec3::new(0.0, 1.0, 0.0),
            test_material(),
        );
        let bbox = plane.bounding_box();
        assert!(bbox.min.x < -1000.0);
        assert!(bbox.max.x > 1000.0);
    }

    // ---- HittableList tests ----

    #[test]
    fn hittable_list_empty() {
        let list = HittableList::new();
        let ray = Ray::new(Point3::new(0.0, 0.0, 0.0), Vec3::new(0.0, 0.0, -1.0));
        assert!(list.hit(&ray, 0.001, f64::INFINITY).is_none());
    }

    #[test]
    fn hittable_list_single_object() {
        let mut list = HittableList::new();
        list.add(Box::new(Sphere::new(
            Point3::new(0.0, 0.0, -5.0),
            1.0,
            test_material(),
        )));
        let ray = Ray::new(Point3::new(0.0, 0.0, 0.0), Vec3::new(0.0, 0.0, -1.0));
        assert!(list.hit(&ray, 0.001, f64::INFINITY).is_some());
    }

    #[test]
    fn hittable_list_closest_hit() {
        let mut list = HittableList::new();
        // Closer sphere
        list.add(Box::new(Sphere::new(
            Point3::new(0.0, 0.0, -3.0),
            0.5,
            test_material(),
        )));
        // Farther sphere
        list.add(Box::new(Sphere::new(
            Point3::new(0.0, 0.0, -10.0),
            0.5,
            test_material(),
        )));
        let ray = Ray::new(Point3::new(0.0, 0.0, 0.0), Vec3::new(0.0, 0.0, -1.0));
        let rec = list.hit(&ray, 0.001, f64::INFINITY).unwrap();
        // Should hit the closer sphere (at t ~ 2.5)
        assert!(rec.t < 5.0);
    }

    #[test]
    fn hittable_list_bounding_box_encloses_all() {
        let mut list = HittableList::new();
        list.add(Box::new(Sphere::new(
            Point3::new(-5.0, 0.0, 0.0),
            1.0,
            test_material(),
        )));
        list.add(Box::new(Sphere::new(
            Point3::new(5.0, 0.0, 0.0),
            1.0,
            test_material(),
        )));
        let bbox = list.bounding_box();
        assert!(bbox.min.x <= -6.0);
        assert!(bbox.max.x >= 6.0);
    }

    // ---- Additional Sphere tests ----

    #[test]
    fn sphere_hit_tangent_ray() {
        // Ray that just barely grazes the sphere (tangent)
        let sphere = Sphere::new(Point3::new(0.0, 1.0, -5.0), 1.0, test_material());
        let ray = Ray::new(Point3::new(0.0, 0.0, 0.0), Vec3::new(0.0, 0.0, -1.0));
        // The ray goes along z-axis at y=0, sphere center is at y=1 with radius 1
        // Distance from ray to center = 1.0 = radius, so tangent
        let hit = sphere.hit(&ray, 0.001, f64::INFINITY);
        // Tangent ray should still hit (discriminant = 0)
        assert!(hit.is_some());
    }

    #[test]
    fn sphere_hit_behind_ray() {
        // Sphere is behind the ray origin
        let sphere = Sphere::new(Point3::new(0.0, 0.0, 5.0), 1.0, test_material());
        let ray = Ray::new(Point3::new(0.0, 0.0, 0.0), Vec3::new(0.0, 0.0, -1.0));
        // Ray goes in -z direction, sphere is at z=5, so behind origin
        let hit = sphere.hit(&ray, 0.001, f64::INFINITY);
        assert!(hit.is_none());
    }

    #[test]
    fn sphere_hit_second_root_in_range() {
        // First root is before t_min, but second root is in range
        let sphere = Sphere::new(Point3::new(0.0, 0.0, 0.0), 2.0, test_material());
        let ray = Ray::new(Point3::new(0.0, 0.0, 0.0), Vec3::new(1.0, 0.0, 0.0));
        // Origin is at center. First root = -2 (behind), second root = +2 (in front).
        // With t_min=0.001, first root is rejected, second is accepted
        let hit = sphere.hit(&ray, 0.001, f64::INFINITY);
        assert!(hit.is_some());
        let rec = hit.unwrap();
        assert!(approx_eq(rec.t, 2.0));
    }

    #[test]
    fn sphere_hit_both_roots_outside_range() {
        // Both roots are outside the t range
        let sphere = Sphere::new(Point3::new(0.0, 0.0, -5.0), 1.0, test_material());
        let ray = Ray::new(Point3::new(0.0, 0.0, 0.0), Vec3::new(0.0, 0.0, -1.0));
        // Roots at t=4 and t=6, but restrict to t=7..10
        let hit = sphere.hit(&ray, 7.0, 10.0);
        assert!(hit.is_none());
    }

    #[test]
    fn sphere_bounding_box_unit_sphere_at_origin() {
        let sphere = Sphere::new(Point3::new(0.0, 0.0, 0.0), 1.0, test_material());
        let bbox = sphere.bounding_box();
        assert!(vec3_approx_eq(bbox.min, Point3::new(-1.0, -1.0, -1.0)));
        assert!(vec3_approx_eq(bbox.max, Point3::new(1.0, 1.0, 1.0)));
    }

    #[test]
    fn sphere_hit_oblique_angle() {
        let sphere = Sphere::new(Point3::new(0.0, 0.0, -5.0), 1.0, test_material());
        let ray = Ray::new(
            Point3::new(0.0, 0.0, 0.0),
            Vec3::new(0.1, 0.0, -1.0).unit(),
        );
        let hit = sphere.hit(&ray, 0.001, f64::INFINITY);
        assert!(hit.is_some());
        let rec = hit.unwrap();
        // Normal should be unit length
        assert!(approx_eq(rec.normal.length(), 1.0));
    }

    // ---- Additional Plane tests ----

    #[test]
    fn plane_hit_t_range_excluded() {
        let plane = Plane::new(
            Point3::new(0.0, 0.0, 0.0),
            Vec3::new(0.0, 1.0, 0.0),
            test_material(),
        );
        let ray = Ray::new(Point3::new(0.0, 5.0, 0.0), Vec3::new(0.0, -1.0, 0.0));
        // Hit would be at t=5, but restrict range to t=0..3
        let hit = plane.hit(&ray, 0.001, 3.0);
        assert!(hit.is_none());
    }

    #[test]
    fn plane_hit_at_oblique_angle() {
        let plane = Plane::new(
            Point3::new(0.0, 0.0, 0.0),
            Vec3::new(0.0, 1.0, 0.0),
            test_material(),
        );
        let ray = Ray::new(
            Point3::new(3.0, 5.0, 0.0),
            Vec3::new(-1.0, -1.0, 0.0).unit(),
        );
        let hit = plane.hit(&ray, 0.001, f64::INFINITY);
        assert!(hit.is_some());
        let rec = hit.unwrap();
        // Hit point should be on the plane (y ≈ 0)
        assert!(approx_eq(rec.point.y, 0.0));
    }

    #[test]
    fn plane_hit_normal_is_consistent() {
        let plane = Plane::new(
            Point3::new(0.0, 0.0, 0.0),
            Vec3::new(0.0, 1.0, 0.0),
            test_material(),
        );
        // Ray from above - front face
        let ray_above = Ray::new(Point3::new(0.0, 5.0, 0.0), Vec3::new(0.0, -1.0, 0.0));
        let rec_above = plane.hit(&ray_above, 0.001, f64::INFINITY).unwrap();
        assert!(rec_above.front_face);
        assert!(approx_eq(rec_above.normal.y, 1.0)); // Normal points up (toward ray)

        // Ray from below - back face
        let ray_below = Ray::new(Point3::new(0.0, -5.0, 0.0), Vec3::new(0.0, 1.0, 0.0));
        let rec_below = plane.hit(&ray_below, 0.001, f64::INFINITY).unwrap();
        assert!(!rec_below.front_face);
        assert!(approx_eq(rec_below.normal.y, -1.0)); // Normal flipped
    }

    #[test]
    fn plane_with_non_axis_aligned_normal() {
        let plane = Plane::new(
            Point3::new(0.0, 0.0, 0.0),
            Vec3::new(1.0, 1.0, 0.0), // 45 degree normal (will be normalized)
            test_material(),
        );
        // Ray perpendicular to the tilted normal
        let ray = Ray::new(Point3::new(5.0, 5.0, 0.0), Vec3::new(-1.0, -1.0, 0.0).unit());
        let hit = plane.hit(&ray, 0.001, f64::INFINITY);
        assert!(hit.is_some());
    }

    // ---- Additional HittableList tests ----

    #[test]
    fn hittable_list_bounding_box_empty() {
        let list = HittableList::new();
        let bbox = list.bounding_box();
        // Empty list should return empty bounding box
        assert_eq!(bbox.min.x, f64::INFINITY);
        assert_eq!(bbox.max.x, f64::NEG_INFINITY);
    }

    #[test]
    fn hittable_list_miss_all() {
        let mut list = HittableList::new();
        list.add(Box::new(Sphere::new(
            Point3::new(100.0, 0.0, 0.0),
            1.0,
            test_material(),
        )));
        list.add(Box::new(Sphere::new(
            Point3::new(-100.0, 0.0, 0.0),
            1.0,
            test_material(),
        )));
        // Ray that misses both spheres
        let ray = Ray::new(Point3::new(0.0, 0.0, 0.0), Vec3::new(0.0, 1.0, 0.0));
        assert!(list.hit(&ray, 0.001, f64::INFINITY).is_none());
    }

    #[test]
    fn hittable_list_multiple_objects_same_direction() {
        let mut list = HittableList::new();
        // Three spheres along the z-axis
        list.add(Box::new(Sphere::new(
            Point3::new(0.0, 0.0, -3.0),
            0.5,
            test_material(),
        )));
        list.add(Box::new(Sphere::new(
            Point3::new(0.0, 0.0, -6.0),
            0.5,
            test_material(),
        )));
        list.add(Box::new(Sphere::new(
            Point3::new(0.0, 0.0, -9.0),
            0.5,
            test_material(),
        )));
        let ray = Ray::new(Point3::new(0.0, 0.0, 0.0), Vec3::new(0.0, 0.0, -1.0));
        let rec = list.hit(&ray, 0.001, f64::INFINITY).unwrap();
        // Should hit closest at z=-3, so t ≈ 2.5
        assert!(rec.t < 4.0);
        assert!(rec.t > 1.0);
    }

    #[test]
    fn hittable_list_with_plane_and_sphere() {
        let mut list = HittableList::new();
        list.add(Box::new(Plane::new(
            Point3::new(0.0, -1.0, 0.0),
            Vec3::new(0.0, 1.0, 0.0),
            test_material(),
        )));
        list.add(Box::new(Sphere::new(
            Point3::new(0.0, 0.0, -5.0),
            1.0,
            test_material(),
        )));

        // Ray that hits the sphere (not the plane)
        let ray = Ray::new(Point3::new(0.0, 0.0, 0.0), Vec3::new(0.0, 0.0, -1.0));
        let rec = list.hit(&ray, 0.001, f64::INFINITY).unwrap();
        // Should hit sphere, not plane (plane is at y=-1, ray goes along z)
        assert!(rec.t < 6.0);
    }

    // ---- HitRecord set_face_normal edge cases ----

    #[test]
    fn hit_record_face_normal_with_diagonal_ray() {
        let ray = Ray::new(
            Point3::new(0.0, 0.0, 0.0),
            Vec3::new(1.0, 1.0, 1.0),
        );
        let outward_normal = Vec3::new(-1.0, 0.0, 0.0);
        let (normal, front_face) = HitRecord::set_face_normal(&ray, outward_normal);
        // direction dot outward_normal = 1.0 * -1.0 = -1.0 < 0 => front face
        assert!(front_face);
        assert!(vec3_approx_eq(normal, outward_normal));
    }
}
