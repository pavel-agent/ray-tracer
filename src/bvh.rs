use rand::Rng;

use crate::hittable::{HitRecord, Hittable};
use crate::ray::{Aabb, Ray};

pub struct BvhNode {
    left: Box<dyn Hittable>,
    right: Box<dyn Hittable>,
    bbox: Aabb,
}

impl BvhNode {
    pub fn new(mut objects: Vec<Box<dyn Hittable>>) -> Self {
        let mut rng = rand::thread_rng();
        let axis = rng.gen_range(0..3_usize);

        let (left, right): (Box<dyn Hittable>, Box<dyn Hittable>) = match objects.len() {
            1 => {
                let a = objects.remove(0);
                let bbox = a.bounding_box();
                // Duplicate as both children
                return Self {
                    bbox,
                    left: a,
                    right: Box::new(EmptyHittable { bbox }),
                };
            }
            2 => {
                let b = objects.remove(1);
                let a = objects.remove(0);
                if box_compare(&*a, &*b, axis) == std::cmp::Ordering::Less {
                    (a, b)
                } else {
                    (b, a)
                }
            }
            _ => {
                objects.sort_by(|a, b| box_compare(&**a, &**b, axis));
                let mid = objects.len() / 2;
                let right_half: Vec<_> = objects.drain(mid..).collect();
                let left_half = objects;
                (
                    Box::new(BvhNode::new(left_half)) as Box<dyn Hittable>,
                    Box::new(BvhNode::new(right_half)) as Box<dyn Hittable>,
                )
            }
        };

        let bbox = Aabb::surrounding_box(left.bounding_box(), right.bounding_box());
        Self { left, right, bbox }
    }
}

impl Hittable for BvhNode {
    fn hit(&self, ray: &Ray, t_min: f64, t_max: f64) -> Option<HitRecord> {
        if !self.bbox.hit(ray, t_min, t_max) {
            return None;
        }

        let hit_left = self.left.hit(ray, t_min, t_max);
        let t_max_right = hit_left.as_ref().map_or(t_max, |rec| rec.t);
        let hit_right = self.right.hit(ray, t_min, t_max_right);

        hit_right.or(hit_left)
    }

    fn bounding_box(&self) -> Aabb {
        self.bbox
    }
}

fn box_compare(a: &dyn Hittable, b: &dyn Hittable, axis: usize) -> std::cmp::Ordering {
    let a_box = a.bounding_box();
    let b_box = b.bounding_box();
    a_box
        .min
        .component(axis)
        .partial_cmp(&b_box.min.component(axis))
        .unwrap_or(std::cmp::Ordering::Equal)
}

// A sentinel empty hittable for single-element BVH leaves
struct EmptyHittable {
    bbox: Aabb,
}

impl Hittable for EmptyHittable {
    fn hit(&self, _ray: &Ray, _t_min: f64, _t_max: f64) -> Option<HitRecord> {
        None
    }

    fn bounding_box(&self) -> Aabb {
        self.bbox
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::sync::Arc;

    use crate::hittable::Sphere;
    use crate::material::Material;
    use crate::ray::{Color, Point3, Vec3};

    fn test_material() -> Arc<Material> {
        Arc::new(Material::Lambertian {
            albedo: Color::new(0.5, 0.5, 0.5),
        })
    }

    fn make_sphere(x: f64, y: f64, z: f64, r: f64) -> Box<dyn Hittable> {
        Box::new(Sphere::new(Point3::new(x, y, z), r, test_material()))
    }

    // ---- BVH construction ----

    #[test]
    fn bvh_single_object() {
        let objects: Vec<Box<dyn Hittable>> = vec![make_sphere(0.0, 0.0, -5.0, 1.0)];
        let bvh = BvhNode::new(objects);
        let bbox = bvh.bounding_box();
        assert!(bbox.min.x <= -1.0);
        assert!(bbox.max.x >= 1.0);
    }

    #[test]
    fn bvh_two_objects() {
        let objects: Vec<Box<dyn Hittable>> = vec![
            make_sphere(-3.0, 0.0, 0.0, 1.0),
            make_sphere(3.0, 0.0, 0.0, 1.0),
        ];
        let bvh = BvhNode::new(objects);
        let bbox = bvh.bounding_box();
        assert!(bbox.min.x <= -4.0);
        assert!(bbox.max.x >= 4.0);
    }

    #[test]
    fn bvh_many_objects() {
        let objects: Vec<Box<dyn Hittable>> = (0..20)
            .map(|i| make_sphere(i as f64 * 3.0, 0.0, 0.0, 1.0))
            .collect();
        let bvh = BvhNode::new(objects);
        // Should not panic and should have a valid bounding box
        let bbox = bvh.bounding_box();
        assert!(bbox.min.x <= -1.0);
        assert!(bbox.max.x >= 58.0); // 19*3 + 1
    }

    // ---- BVH ray intersection ----

    #[test]
    fn bvh_hit_single_sphere() {
        let objects: Vec<Box<dyn Hittable>> = vec![make_sphere(0.0, 0.0, -5.0, 1.0)];
        let bvh = BvhNode::new(objects);
        let ray = Ray::new(Point3::new(0.0, 0.0, 0.0), Vec3::new(0.0, 0.0, -1.0));
        let hit = bvh.hit(&ray, 0.001, f64::INFINITY);
        assert!(hit.is_some());
    }

    #[test]
    fn bvh_miss() {
        let objects: Vec<Box<dyn Hittable>> = vec![make_sphere(0.0, 0.0, -5.0, 1.0)];
        let bvh = BvhNode::new(objects);
        let ray = Ray::new(Point3::new(0.0, 0.0, 0.0), Vec3::new(0.0, 1.0, 0.0));
        let hit = bvh.hit(&ray, 0.001, f64::INFINITY);
        assert!(hit.is_none());
    }

    #[test]
    fn bvh_finds_closest_hit() {
        let objects: Vec<Box<dyn Hittable>> = vec![
            make_sphere(0.0, 0.0, -3.0, 0.5),
            make_sphere(0.0, 0.0, -10.0, 0.5),
        ];
        let bvh = BvhNode::new(objects);
        let ray = Ray::new(Point3::new(0.0, 0.0, 0.0), Vec3::new(0.0, 0.0, -1.0));
        let rec = bvh.hit(&ray, 0.001, f64::INFINITY).unwrap();
        // Closer sphere is at z=-3 with r=0.5, so hit at t~2.5
        assert!(rec.t < 5.0);
    }

    #[test]
    fn bvh_hit_respects_t_range() {
        let objects: Vec<Box<dyn Hittable>> = vec![make_sphere(0.0, 0.0, -5.0, 1.0)];
        let bvh = BvhNode::new(objects);
        let ray = Ray::new(Point3::new(0.0, 0.0, 0.0), Vec3::new(0.0, 0.0, -1.0));
        // Sphere is at t=4..6, so t_max=3 should miss
        let hit = bvh.hit(&ray, 0.001, 3.0);
        assert!(hit.is_none());
    }

    #[test]
    fn bvh_many_spheres_hit() {
        // Place many spheres along a line and hit one of them
        let objects: Vec<Box<dyn Hittable>> = (0..10)
            .map(|i| make_sphere(i as f64 * 5.0, 0.0, -5.0, 1.0))
            .collect();
        let bvh = BvhNode::new(objects);

        // Ray aimed at the first sphere (at x=0)
        let ray = Ray::new(Point3::new(0.0, 0.0, 0.0), Vec3::new(0.0, 0.0, -1.0));
        assert!(bvh.hit(&ray, 0.001, f64::INFINITY).is_some());

        // Ray aimed at the last sphere (at x=45)
        let ray2 = Ray::new(Point3::new(45.0, 0.0, 0.0), Vec3::new(0.0, 0.0, -1.0));
        assert!(bvh.hit(&ray2, 0.001, f64::INFINITY).is_some());

        // Ray that misses everything
        let ray3 = Ray::new(Point3::new(100.0, 100.0, 0.0), Vec3::new(0.0, 0.0, -1.0));
        assert!(bvh.hit(&ray3, 0.001, f64::INFINITY).is_none());
    }

    #[test]
    fn bvh_bounding_box_encloses_all_objects() {
        let objects: Vec<Box<dyn Hittable>> = vec![
            make_sphere(-10.0, -10.0, -10.0, 1.0),
            make_sphere(10.0, 10.0, 10.0, 1.0),
        ];
        let bvh = BvhNode::new(objects);
        let bbox = bvh.bounding_box();
        assert!(bbox.min.x <= -11.0);
        assert!(bbox.min.y <= -11.0);
        assert!(bbox.min.z <= -11.0);
        assert!(bbox.max.x >= 11.0);
        assert!(bbox.max.y >= 11.0);
        assert!(bbox.max.z >= 11.0);
    }

    // ---- box_compare ----

    #[test]
    fn box_compare_by_axis() {
        let a = Sphere::new(Point3::new(-5.0, 0.0, 0.0), 1.0, test_material());
        let b = Sphere::new(Point3::new(5.0, 0.0, 0.0), 1.0, test_material());

        // Compare along x-axis: a should be less than b
        assert_eq!(box_compare(&a, &b, 0), std::cmp::Ordering::Less);
        assert_eq!(box_compare(&b, &a, 0), std::cmp::Ordering::Greater);
    }

    #[test]
    fn box_compare_equal() {
        let a = Sphere::new(Point3::new(0.0, 0.0, 0.0), 1.0, test_material());
        let b = Sphere::new(Point3::new(0.0, 5.0, 5.0), 1.0, test_material());

        // Same x min, so equal along axis 0
        assert_eq!(box_compare(&a, &b, 0), std::cmp::Ordering::Equal);
    }

    // ---- Additional BVH tests ----

    #[test]
    fn bvh_three_objects() {
        // Tests the recursive split path (len >= 3)
        let objects: Vec<Box<dyn Hittable>> = vec![
            make_sphere(-5.0, 0.0, 0.0, 1.0),
            make_sphere(0.0, 0.0, 0.0, 1.0),
            make_sphere(5.0, 0.0, 0.0, 1.0),
        ];
        let bvh = BvhNode::new(objects);
        let bbox = bvh.bounding_box();
        assert!(bbox.min.x <= -6.0);
        assert!(bbox.max.x >= 6.0);
    }

    #[test]
    fn bvh_hit_each_of_three_spheres() {
        let objects: Vec<Box<dyn Hittable>> = vec![
            make_sphere(-5.0, 0.0, -5.0, 1.0),
            make_sphere(0.0, 0.0, -5.0, 1.0),
            make_sphere(5.0, 0.0, -5.0, 1.0),
        ];
        let bvh = BvhNode::new(objects);

        // Hit left sphere
        let ray1 = Ray::new(Point3::new(-5.0, 0.0, 0.0), Vec3::new(0.0, 0.0, -1.0));
        assert!(bvh.hit(&ray1, 0.001, f64::INFINITY).is_some());

        // Hit center sphere
        let ray2 = Ray::new(Point3::new(0.0, 0.0, 0.0), Vec3::new(0.0, 0.0, -1.0));
        assert!(bvh.hit(&ray2, 0.001, f64::INFINITY).is_some());

        // Hit right sphere
        let ray3 = Ray::new(Point3::new(5.0, 0.0, 0.0), Vec3::new(0.0, 0.0, -1.0));
        assert!(bvh.hit(&ray3, 0.001, f64::INFINITY).is_some());
    }

    #[test]
    fn bvh_single_object_hit_and_miss() {
        let objects: Vec<Box<dyn Hittable>> = vec![make_sphere(0.0, 0.0, -5.0, 1.0)];
        let bvh = BvhNode::new(objects);

        // Hit
        let ray_hit = Ray::new(Point3::new(0.0, 0.0, 0.0), Vec3::new(0.0, 0.0, -1.0));
        assert!(bvh.hit(&ray_hit, 0.001, f64::INFINITY).is_some());

        // Miss (EmptyHittable path)
        let ray_miss = Ray::new(Point3::new(0.0, 100.0, 0.0), Vec3::new(0.0, 0.0, -1.0));
        assert!(bvh.hit(&ray_miss, 0.001, f64::INFINITY).is_none());
    }

    #[test]
    fn bvh_bbox_miss_early_exit() {
        // A ray that misses the BVH bounding box entirely should return None quickly
        let objects: Vec<Box<dyn Hittable>> = vec![
            make_sphere(0.0, 0.0, -5.0, 1.0),
            make_sphere(2.0, 0.0, -5.0, 1.0),
        ];
        let bvh = BvhNode::new(objects);
        // Ray pointing away from all objects
        let ray = Ray::new(Point3::new(0.0, 100.0, 100.0), Vec3::new(0.0, 1.0, 1.0));
        assert!(bvh.hit(&ray, 0.001, f64::INFINITY).is_none());
    }

    #[test]
    fn bvh_overlapping_spheres() {
        // Overlapping spheres - should still find closest hit
        let objects: Vec<Box<dyn Hittable>> = vec![
            make_sphere(0.0, 0.0, -3.0, 2.0),
            make_sphere(0.0, 0.0, -5.0, 2.0),
        ];
        let bvh = BvhNode::new(objects);
        let ray = Ray::new(Point3::new(0.0, 0.0, 0.0), Vec3::new(0.0, 0.0, -1.0));
        let rec = bvh.hit(&ray, 0.001, f64::INFINITY).unwrap();
        // Should hit the closer sphere first (at t=1.0, z=-1 is the front of the first sphere)
        assert!(rec.t < 3.0);
    }

    #[test]
    fn box_compare_by_y_axis() {
        let a = Sphere::new(Point3::new(0.0, -5.0, 0.0), 1.0, test_material());
        let b = Sphere::new(Point3::new(0.0, 5.0, 0.0), 1.0, test_material());
        assert_eq!(box_compare(&a, &b, 1), std::cmp::Ordering::Less);
        assert_eq!(box_compare(&b, &a, 1), std::cmp::Ordering::Greater);
    }

    #[test]
    fn box_compare_by_z_axis() {
        let a = Sphere::new(Point3::new(0.0, 0.0, -5.0), 1.0, test_material());
        let b = Sphere::new(Point3::new(0.0, 0.0, 5.0), 1.0, test_material());
        assert_eq!(box_compare(&a, &b, 2), std::cmp::Ordering::Less);
        assert_eq!(box_compare(&b, &a, 2), std::cmp::Ordering::Greater);
    }

    #[test]
    fn bvh_large_scene() {
        // Stress test with many objects
        let objects: Vec<Box<dyn Hittable>> = (0..100)
            .map(|i| {
                let x = (i % 10) as f64 * 3.0;
                let z = (i / 10) as f64 * 3.0;
                make_sphere(x, 0.0, -z, 0.5)
            })
            .collect();
        let bvh = BvhNode::new(objects);

        // Hit a specific sphere
        let ray = Ray::new(Point3::new(0.0, 0.0, 5.0), Vec3::new(0.0, 0.0, -1.0));
        assert!(bvh.hit(&ray, 0.001, f64::INFINITY).is_some());

        // Miss everything
        let ray_miss = Ray::new(Point3::new(-100.0, 0.0, 0.0), Vec3::new(0.0, 1.0, 0.0));
        assert!(bvh.hit(&ray_miss, 0.001, f64::INFINITY).is_none());
    }

    #[test]
    fn bvh_two_objects_ordering() {
        // Test that two objects are properly ordered regardless of input order
        let objects_a: Vec<Box<dyn Hittable>> = vec![
            make_sphere(-5.0, 0.0, -5.0, 1.0),
            make_sphere(5.0, 0.0, -5.0, 1.0),
        ];
        let bvh_a = BvhNode::new(objects_a);

        let objects_b: Vec<Box<dyn Hittable>> = vec![
            make_sphere(5.0, 0.0, -5.0, 1.0),
            make_sphere(-5.0, 0.0, -5.0, 1.0),
        ];
        let bvh_b = BvhNode::new(objects_b);

        // Both should have the same bounding box
        let bbox_a = bvh_a.bounding_box();
        let bbox_b = bvh_b.bounding_box();
        assert!(bbox_a.min.x <= -6.0 && bbox_b.min.x <= -6.0);
        assert!(bbox_a.max.x >= 6.0 && bbox_b.max.x >= 6.0);

        // Both should hit the same ray
        let ray = Ray::new(Point3::new(-5.0, 0.0, 0.0), Vec3::new(0.0, 0.0, -1.0));
        assert!(bvh_a.hit(&ray, 0.001, f64::INFINITY).is_some());
        assert!(bvh_b.hit(&ray, 0.001, f64::INFINITY).is_some());
    }

    #[test]
    fn empty_hittable_never_hits() {
        let bbox = crate::ray::Aabb::new(
            Point3::new(-1.0, -1.0, -1.0),
            Point3::new(1.0, 1.0, 1.0),
        );
        let empty = EmptyHittable { bbox };
        let ray = Ray::new(Point3::new(0.0, 0.0, 0.0), Vec3::new(0.0, 0.0, -1.0));
        assert!(empty.hit(&ray, 0.001, f64::INFINITY).is_none());
    }

    #[test]
    fn empty_hittable_returns_bbox() {
        let bbox = crate::ray::Aabb::new(
            Point3::new(-2.0, -3.0, -4.0),
            Point3::new(2.0, 3.0, 4.0),
        );
        let empty = EmptyHittable { bbox };
        let returned_bbox = empty.bounding_box();
        assert!((returned_bbox.min.x - (-2.0)).abs() < 1e-10);
        assert!((returned_bbox.max.z - 4.0).abs() < 1e-10);
    }
}
