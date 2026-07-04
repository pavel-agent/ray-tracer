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
