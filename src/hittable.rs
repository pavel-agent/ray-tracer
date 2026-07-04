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
