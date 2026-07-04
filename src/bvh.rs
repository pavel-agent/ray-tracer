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
