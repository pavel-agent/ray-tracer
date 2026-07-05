mod bvh;
mod camera;
mod hittable;
mod material;
mod ray;
mod scene;

use std::io::Write;

use clap::Parser;
use rand::Rng;
use rayon::prelude::*;

use bvh::BvhNode;
use hittable::Hittable;
use ray::{Color, Ray};

#[derive(Parser)]
#[command(name = "ray-tracer", about = "A Rust path tracer")]
struct Cli {
    /// Image width in pixels
    #[arg(short = 'W', long, default_value_t = 800)]
    width: u32,

    /// Image height in pixels
    #[arg(short = 'H', long, default_value_t = 450)]
    height: u32,

    /// Samples per pixel
    #[arg(short, long, default_value_t = 100)]
    samples: u32,

    /// Maximum ray bounce depth
    #[arg(short, long, default_value_t = 50)]
    depth: u32,

    /// Output file path (.png or .ppm)
    #[arg(short, long, default_value = "output.png")]
    output: String,
}

fn ray_color(ray: &Ray, world: &dyn Hittable, depth: u32, rng: &mut impl Rng) -> Color {
    if depth == 0 {
        return Color::new(0.0, 0.0, 0.0);
    }

    if let Some(rec) = world.hit(ray, 0.001, f64::INFINITY) {
        if let Some(scatter) = rec.material.scatter(ray, &rec, rng) {
            return scatter.attenuation * ray_color(&scatter.scattered, world, depth - 1, rng);
        }
        return Color::new(0.0, 0.0, 0.0);
    }

    // Sky gradient
    let unit_dir = ray.direction.unit();
    let t = 0.5 * (unit_dir.y + 1.0);
    Color::new(1.0, 1.0, 1.0) * (1.0 - t) + Color::new(0.5, 0.7, 1.0) * t
}

fn clamp(x: f64, min: f64, max: f64) -> f64 {
    x.max(min).min(max)
}

fn main() {
    let cli = Cli::parse();

    let width = cli.width;
    let height = cli.height;
    let samples = cli.samples;
    let max_depth = cli.depth;
    let aspect_ratio = width as f64 / height as f64;

    eprintln!(
        "Rendering {}x{} image, {} samples/pixel, max depth {}",
        width, height, samples, max_depth
    );

    let (camera, objects) = scene::build_demo_scene(aspect_ratio);

    // Build BVH
    let world = BvhNode::new(objects);

    // Render rows in parallel
    let pixels: Vec<Vec<Color>> = (0..height)
        .into_par_iter()
        .rev()
        .map(|j| {
            let mut rng = rand::thread_rng();
            let mut row = Vec::with_capacity(width as usize);
            for i in 0..width {
                let mut pixel_color = Color::new(0.0, 0.0, 0.0);
                for _ in 0..samples {
                    let u = (i as f64 + rng.gen::<f64>()) / (width - 1) as f64;
                    let v = (j as f64 + rng.gen::<f64>()) / (height - 1) as f64;
                    let r = camera.get_ray(u, v, &mut rng);
                    pixel_color += ray_color(&r, &world, max_depth, &mut rng);
                }
                row.push(pixel_color);
            }
            if j % 50 == 0 {
                eprintln!("  scanline {} done", j);
            }
            row
        })
        .collect();

    // Write output
    if cli.output.ends_with(".ppm") {
        write_ppm(&cli.output, width, height, samples, &pixels);
    } else {
        write_png(&cli.output, width, height, samples, &pixels);
    }

    eprintln!("Done. Wrote {}", cli.output);
}

fn write_ppm(path: &str, width: u32, height: u32, samples: u32, pixels: &[Vec<Color>]) {
    let mut file = std::fs::File::create(path).expect("Failed to create PPM file");
    write!(file, "P3\n{} {}\n255\n", width, height).unwrap();

    let scale = 1.0 / samples as f64;
    for row in pixels {
        for color in row {
            let r = (clamp((color.x * scale).sqrt(), 0.0, 0.999) * 256.0) as u32;
            let g = (clamp((color.y * scale).sqrt(), 0.0, 0.999) * 256.0) as u32;
            let b = (clamp((color.z * scale).sqrt(), 0.0, 0.999) * 256.0) as u32;
            write!(file, "{} {} {}\n", r, g, b).unwrap();
        }
    }
}

fn write_png(path: &str, width: u32, height: u32, samples: u32, pixels: &[Vec<Color>]) {
    let mut img = image::RgbImage::new(width, height);
    let scale = 1.0 / samples as f64;

    for (y, row) in pixels.iter().enumerate() {
        for (x, color) in row.iter().enumerate() {
            let r = (clamp((color.x * scale).sqrt(), 0.0, 0.999) * 256.0) as u8;
            let g = (clamp((color.y * scale).sqrt(), 0.0, 0.999) * 256.0) as u8;
            let b = (clamp((color.z * scale).sqrt(), 0.0, 0.999) * 256.0) as u8;
            img.put_pixel(x as u32, y as u32, image::Rgb([r, g, b]));
        }
    }

    img.save(path).expect("Failed to save PNG");
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::sync::Arc;
    use hittable::{HittableList, Sphere};
    use material::Material;
    use ray::{Point3, Vec3};

    const EPSILON: f64 = 1e-6;

    fn approx_eq(a: f64, b: f64) -> bool {
        (a - b).abs() < EPSILON
    }

    fn vec3_approx_eq(a: Vec3, b: Vec3) -> bool {
        approx_eq(a.x, b.x) && approx_eq(a.y, b.y) && approx_eq(a.z, b.z)
    }

    fn test_material() -> Arc<Material> {
        Arc::new(Material::Lambertian {
            albedo: Color::new(0.5, 0.5, 0.5),
        })
    }

    // ---- clamp tests ----

    #[test]
    fn clamp_within_range() {
        assert!(approx_eq(clamp(0.5, 0.0, 1.0), 0.5));
    }

    #[test]
    fn clamp_below_min() {
        assert!(approx_eq(clamp(-1.0, 0.0, 1.0), 0.0));
    }

    #[test]
    fn clamp_above_max() {
        assert!(approx_eq(clamp(2.0, 0.0, 1.0), 1.0));
    }

    #[test]
    fn clamp_at_min_boundary() {
        assert!(approx_eq(clamp(0.0, 0.0, 1.0), 0.0));
    }

    #[test]
    fn clamp_at_max_boundary() {
        assert!(approx_eq(clamp(1.0, 0.0, 1.0), 1.0));
    }

    #[test]
    fn clamp_negative_range() {
        assert!(approx_eq(clamp(-0.5, -1.0, 0.0), -0.5));
        assert!(approx_eq(clamp(-2.0, -1.0, 0.0), -1.0));
        assert!(approx_eq(clamp(1.0, -1.0, 0.0), 0.0));
    }

    // ---- ray_color tests ----

    #[test]
    fn ray_color_returns_black_at_depth_zero() {
        let mut rng = rand::thread_rng();
        let world = HittableList::new();
        let ray = Ray::new(Point3::new(0.0, 0.0, 0.0), Vec3::new(0.0, 0.0, -1.0));
        let color = ray_color(&ray, &world, 0, &mut rng);
        assert!(vec3_approx_eq(color, Color::new(0.0, 0.0, 0.0)));
    }

    #[test]
    fn ray_color_returns_sky_when_no_hit() {
        let mut rng = rand::thread_rng();
        let world = HittableList::new();
        // Ray pointing straight up
        let ray = Ray::new(Point3::new(0.0, 0.0, 0.0), Vec3::new(0.0, 1.0, 0.0));
        let color = ray_color(&ray, &world, 10, &mut rng);
        // unit_dir.y = 1.0, t = 0.5 * (1.0 + 1.0) = 1.0
        // color = white * 0.0 + blue * 1.0 = (0.5, 0.7, 1.0)
        assert!(approx_eq(color.x, 0.5));
        assert!(approx_eq(color.y, 0.7));
        assert!(approx_eq(color.z, 1.0));
    }

    #[test]
    fn ray_color_returns_white_for_straight_down() {
        let mut rng = rand::thread_rng();
        let world = HittableList::new();
        // Ray pointing straight down
        let ray = Ray::new(Point3::new(0.0, 0.0, 0.0), Vec3::new(0.0, -1.0, 0.0));
        let color = ray_color(&ray, &world, 10, &mut rng);
        // unit_dir.y = -1.0, t = 0.5 * (-1.0 + 1.0) = 0.0
        // color = white * 1.0 + blue * 0.0 = (1.0, 1.0, 1.0)
        assert!(approx_eq(color.x, 1.0));
        assert!(approx_eq(color.y, 1.0));
        assert!(approx_eq(color.z, 1.0));
    }

    #[test]
    fn ray_color_sky_gradient_horizontal() {
        let mut rng = rand::thread_rng();
        let world = HittableList::new();
        // Ray pointing horizontal (y=0), unit_dir.y = 0.0, t = 0.5
        let ray = Ray::new(Point3::new(0.0, 0.0, 0.0), Vec3::new(0.0, 0.0, -1.0));
        let color = ray_color(&ray, &world, 10, &mut rng);
        // t = 0.5 * (0.0 + 1.0) = 0.5
        // color = white * 0.5 + (0.5, 0.7, 1.0) * 0.5 = (0.75, 0.85, 1.0)
        assert!(approx_eq(color.x, 0.75));
        assert!(approx_eq(color.y, 0.85));
        assert!(approx_eq(color.z, 1.0));
    }

    #[test]
    fn ray_color_hits_sphere() {
        let mut rng = rand::thread_rng();
        let mut world = HittableList::new();
        world.add(Box::new(Sphere::new(
            Point3::new(0.0, 0.0, -5.0),
            1.0,
            test_material(),
        )));
        let ray = Ray::new(Point3::new(0.0, 0.0, 0.0), Vec3::new(0.0, 0.0, -1.0));
        let color = ray_color(&ray, &world, 5, &mut rng);
        // Should not be sky color -- it should be a scattered color
        // The lambertian material will scatter, so we just check it's not pure sky
        let sky_color = Color::new(0.75, 0.85, 1.0);
        assert!(
            !vec3_approx_eq(color, sky_color),
            "Color should differ from sky when hitting a sphere"
        );
    }

    #[test]
    fn ray_color_misses_sphere() {
        let mut rng = rand::thread_rng();
        let mut world = HittableList::new();
        world.add(Box::new(Sphere::new(
            Point3::new(100.0, 100.0, -5.0),
            1.0,
            test_material(),
        )));
        let ray = Ray::new(Point3::new(0.0, 0.0, 0.0), Vec3::new(0.0, 0.0, -1.0));
        let color = ray_color(&ray, &world, 5, &mut rng);
        // Should be sky color since we miss the sphere
        let expected = Color::new(0.75, 0.85, 1.0);
        assert!(vec3_approx_eq(color, expected));
    }

    // ---- write_ppm tests ----

    #[test]
    fn write_ppm_creates_valid_file() {
        let dir = std::env::temp_dir();
        let path = dir.join("test_output.ppm");
        let path_str = path.to_str().unwrap();

        let pixels = vec![
            vec![Color::new(1.0, 0.0, 0.0), Color::new(0.0, 1.0, 0.0)],
            vec![Color::new(0.0, 0.0, 1.0), Color::new(1.0, 1.0, 1.0)],
        ];
        write_ppm(path_str, 2, 2, 1, &pixels);

        let content = std::fs::read_to_string(&path).unwrap();
        assert!(content.starts_with("P3\n2 2\n255\n"));
        // Clean up
        std::fs::remove_file(&path).ok();
    }

    #[test]
    fn write_ppm_gamma_correction() {
        let dir = std::env::temp_dir();
        let path = dir.join("test_gamma.ppm");
        let path_str = path.to_str().unwrap();

        // With 1 sample, a pixel value of 0.25 should be sqrt(0.25) = 0.5 after gamma
        // 0.5 * 256 = 128
        let pixels = vec![vec![Color::new(0.25, 0.25, 0.25)]];
        write_ppm(path_str, 1, 1, 1, &pixels);

        let content = std::fs::read_to_string(&path).unwrap();
        assert!(content.contains("128 128 128"), "PPM should have gamma-corrected values, got: {}", content);
        std::fs::remove_file(&path).ok();
    }

    #[test]
    fn write_ppm_multisampled() {
        let dir = std::env::temp_dir();
        let path = dir.join("test_multisample.ppm");
        let path_str = path.to_str().unwrap();

        // 4 samples, accumulated color = (4.0, 4.0, 4.0), scale = 0.25, scaled = 1.0
        // sqrt(1.0) = 1.0, clamped to 0.999, * 256 = 255
        let pixels = vec![vec![Color::new(4.0, 4.0, 4.0)]];
        write_ppm(path_str, 1, 1, 4, &pixels);

        let content = std::fs::read_to_string(&path).unwrap();
        assert!(content.contains("255 255 255"), "Max value should clamp to 255, got: {}", content);
        std::fs::remove_file(&path).ok();
    }

    // ---- write_png tests ----

    #[test]
    fn write_png_creates_valid_file() {
        let dir = std::env::temp_dir();
        let path = dir.join("test_output.png");
        let path_str = path.to_str().unwrap();

        let pixels = vec![
            vec![Color::new(1.0, 0.0, 0.0), Color::new(0.0, 1.0, 0.0)],
            vec![Color::new(0.0, 0.0, 1.0), Color::new(1.0, 1.0, 1.0)],
        ];
        write_png(path_str, 2, 2, 1, &pixels);

        // Verify the file exists and is a valid PNG
        let img = image::open(&path).expect("Should be a valid PNG");
        assert_eq!(img.width(), 2);
        assert_eq!(img.height(), 2);
        std::fs::remove_file(&path).ok();
    }

    #[test]
    fn write_png_pixel_values() {
        let dir = std::env::temp_dir();
        let path = dir.join("test_pixel_values.png");
        let path_str = path.to_str().unwrap();

        // Pure red pixel (1 sample)
        let pixels = vec![vec![Color::new(1.0, 0.0, 0.0)]];
        write_png(path_str, 1, 1, 1, &pixels);

        let img = image::open(&path).unwrap().to_rgb8();
        let pixel = img.get_pixel(0, 0);
        // sqrt(1.0) = 1.0, clamped to 0.999, * 256 = 255
        assert_eq!(pixel[0], 255); // Red
        assert_eq!(pixel[1], 0);   // Green
        assert_eq!(pixel[2], 0);   // Blue
        std::fs::remove_file(&path).ok();
    }
}
