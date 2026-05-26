//! Chart/graph plotting tool — generates SVG/PNG files using plotters.

use std::path::PathBuf;
use std::time::{SystemTime, UNIX_EPOCH};

use anyhow::{Result, anyhow};
use async_trait::async_trait;
use plotters::prelude::*;
use serde::Deserialize;
use serde_json::{Value, json};

use flashmind_types::tool::{Tool, ToolContext, ToolResult};

// ---------------------------------------------------------------------------
// Types
// ---------------------------------------------------------------------------

#[derive(Deserialize)]
struct PlotArgs {
    chart_type: ChartType,
    title: Option<String>,
    x_label: Option<String>,
    y_label: Option<String>,
    series: Vec<Series>,
    width: Option<u32>,
    height: Option<u32>,
    format: Option<OutputFormat>,
    x_range: Option<[f64; 2]>,
    y_range: Option<[f64; 2]>,
    categories: Option<Vec<String>>,
}

#[derive(Deserialize, Clone, Copy)]
#[serde(rename_all = "snake_case")]
enum ChartType {
    Line,
    Bar,
    Scatter,
    Histogram,
    Area,
}

#[derive(Deserialize, Clone, Copy)]
#[serde(rename_all = "snake_case")]
enum OutputFormat {
    Svg,
    Png,
}

#[derive(Deserialize)]
struct Series {
    name: String,
    x: Option<Vec<f64>>,
    y: Vec<f64>,
    color: Option<String>,
}

// ---------------------------------------------------------------------------
// Tool
// ---------------------------------------------------------------------------

pub struct PlotTool {
    output_dir: PathBuf,
}

impl PlotTool {
    pub fn new(output_dir: PathBuf) -> Self {
        Self { output_dir }
    }
}

#[async_trait]
impl Tool for PlotTool {
    fn name(&self) -> &str {
        "plot"
    }

    fn description(&self) -> &str {
        "Generate a chart/graph as an SVG or PNG file. Supports line, bar, scatter, histogram, and area charts with multiple data series."
    }

    fn parameters(&self) -> Value {
        json!({
            "type": "object",
            "properties": {
                "chart_type": {
                    "type": "string",
                    "enum": ["line", "bar", "scatter", "histogram", "area"],
                    "description": "Type of chart to render"
                },
                "title": {
                    "type": "string",
                    "description": "Chart title displayed at the top"
                },
                "x_label": { "type": "string", "description": "X-axis label" },
                "y_label": { "type": "string", "description": "Y-axis label" },
                "series": {
                    "type": "array",
                    "description": "Data series to plot",
                    "items": {
                        "type": "object",
                        "properties": {
                            "name": { "type": "string", "description": "Legend label" },
                            "x": {
                                "type": "array",
                                "items": { "type": "number" },
                                "description": "X values (indices used if omitted)"
                            },
                            "y": {
                                "type": "array",
                                "items": { "type": "number" },
                                "description": "Y values"
                            },
                            "color": {
                                "type": "string",
                                "description": "Color as hex (#ff0000) or name (red, blue, green, etc.)"
                            }
                        },
                        "required": ["name", "y"]
                    }
                },
                "width": { "type": "integer", "description": "Chart width in pixels (default 800)" },
                "height": { "type": "integer", "description": "Chart height in pixels (default 600)" },
                "format": {
                    "type": "string",
                    "enum": ["svg", "png"],
                    "description": "Output format (default svg)"
                },
                "x_range": {
                    "type": "array",
                    "items": { "type": "number" },
                    "minItems": 2,
                    "maxItems": 2,
                    "description": "Explicit [min, max] for x-axis"
                },
                "y_range": {
                    "type": "array",
                    "items": { "type": "number" },
                    "minItems": 2,
                    "maxItems": 2,
                    "description": "Explicit [min, max] for y-axis"
                },
                "categories": {
                    "type": "array",
                    "items": { "type": "string" },
                    "description": "Category labels for bar chart x-axis"
                }
            },
            "required": ["chart_type", "series"]
        })
    }

    async fn execute(&self, ctx: ToolContext<'_>) -> Result<ToolResult> {
        let args: PlotArgs = ctx.parse_args(self.name())?;

        if args.series.is_empty() {
            return Ok(ToolResult::failure(
                ctx.tool_call_id,
                "At least one series is required",
            ));
        }

        let output_dir = self.output_dir.clone();
        let path = tokio::task::spawn_blocking(move || render_chart(&args, &output_dir)).await??;

        let filename = path.file_name().unwrap_or_default().to_string_lossy();
        let size = std::fs::metadata(&path).map(|m| m.len()).unwrap_or(0);
        let size_kb = size / 1024;

        Ok(ToolResult::success(
            ctx.tool_call_id,
            format!(
                "Chart saved: {} ({size_kb}KB)\nInclude [{}]({}) in your response to show it.",
                path.display(),
                filename,
                path.display()
            ),
        ))
    }

    fn humanize(&self, args: &Value) -> String {
        let chart_type = args
            .get("chart_type")
            .and_then(|v| v.as_str())
            .unwrap_or("chart");
        let title = args.get("title").and_then(|v| v.as_str()).unwrap_or("");
        if title.is_empty() {
            format!("Plotting {chart_type} chart")
        } else {
            format!("Plotting {chart_type} chart: {title}")
        }
    }
}

// ---------------------------------------------------------------------------
// Rendering
// ---------------------------------------------------------------------------

fn render_chart(args: &PlotArgs, output_dir: &PathBuf) -> Result<PathBuf> {
    let width = args.width.unwrap_or(800);
    let height = args.height.unwrap_or(600);
    let format = args.format.unwrap_or(OutputFormat::Svg);

    let timestamp = SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .unwrap()
        .as_millis();

    let ext = match format {
        OutputFormat::Svg => "svg",
        OutputFormat::Png => "png",
    };
    let filename = format!("chart_{timestamp}.{ext}");
    let path = output_dir.join(&filename);

    std::fs::create_dir_all(output_dir).ok();

    match format {
        OutputFormat::Svg => {
            let root = SVGBackend::new(&path, (width, height)).into_drawing_area();
            draw_on_area(&root, args)?;
            root.present()
                .map_err(|e| anyhow!("Failed to write SVG: {e}"))?;
        }
        OutputFormat::Png => {
            let root = BitMapBackend::new(&path, (width, height)).into_drawing_area();
            draw_on_area(&root, args)?;
            root.present()
                .map_err(|e| anyhow!("Failed to write PNG: {e}"))?;
        }
    }

    Ok(path)
}

fn draw_on_area<DB: DrawingBackend>(
    root: &DrawingArea<DB, plotters::coord::Shift>,
    args: &PlotArgs,
) -> Result<()>
where
    DB::ErrorType: 'static,
{
    root.fill(&WHITE).map_err(|e| anyhow!("Fill error: {e}"))?;

    let title = args.title.as_deref().unwrap_or("");
    let x_label = args.x_label.as_deref().unwrap_or("");
    let y_label = args.y_label.as_deref().unwrap_or("");

    match args.chart_type {
        ChartType::Bar => draw_bar_chart(root, args, title, x_label, y_label),
        ChartType::Histogram => draw_histogram(root, args, title, x_label, y_label),
        _ => draw_cartesian_chart(root, args, title, x_label, y_label),
    }
}

// Line, scatter, area charts all use Cartesian2D<f64, f64>
fn draw_cartesian_chart<DB: DrawingBackend>(
    root: &DrawingArea<DB, plotters::coord::Shift>,
    args: &PlotArgs,
    title: &str,
    x_label: &str,
    y_label: &str,
) -> Result<()>
where
    DB::ErrorType: 'static,
{
    let (x_range, y_range) = compute_ranges(&args.series, args.x_range, args.y_range);

    let mut builder = ChartBuilder::on(root);
    if !title.is_empty() {
        builder.caption(title, ("sans-serif", 24).into_font());
    }
    let mut chart = builder
        .margin(15)
        .x_label_area_size(40)
        .y_label_area_size(50)
        .build_cartesian_2d(x_range.clone(), y_range.clone())
        .map_err(|e| anyhow!("Chart build error: {e}"))?;

    let mut mesh = chart.configure_mesh();
    if !x_label.is_empty() {
        mesh.x_desc(x_label);
    }
    if !y_label.is_empty() {
        mesh.y_desc(y_label);
    }
    mesh.draw().map_err(|e| anyhow!("Mesh draw error: {e}"))?;

    for (i, s) in args.series.iter().enumerate() {
        let color = parse_color(s.color.as_deref(), i);
        let x_vals: Vec<f64> =
            s.x.clone()
                .unwrap_or_else(|| (0..s.y.len()).map(|i| i as f64).collect());

        let points: Vec<(f64, f64)> = x_vals
            .iter()
            .zip(s.y.iter())
            .map(|(&x, &y)| (x, y))
            .collect();

        match args.chart_type {
            ChartType::Line => {
                chart
                    .draw_series(LineSeries::new(points.clone(), color.stroke_width(2)))
                    .map_err(|e| anyhow!("Draw error: {e}"))?
                    .label(&s.name)
                    .legend(move |(x, y)| {
                        PathElement::new(vec![(x, y), (x + 20, y)], color.stroke_width(2))
                    });
            }
            ChartType::Scatter => {
                chart
                    .draw_series(
                        points
                            .iter()
                            .map(|&(x, y)| Circle::new((x, y), 4, color.filled())),
                    )
                    .map_err(|e| anyhow!("Draw error: {e}"))?
                    .label(&s.name)
                    .legend(move |(x, y)| Circle::new((x, y), 4, color.filled()));
            }
            ChartType::Area => {
                chart
                    .draw_series(AreaSeries::new(
                        points.clone(),
                        y_range.start,
                        color.mix(0.3).filled(),
                    ))
                    .map_err(|e| anyhow!("Draw error: {e}"))?
                    .label(&s.name)
                    .legend(move |(x, y)| {
                        Rectangle::new([(x, y - 5), (x + 20, y + 5)], color.mix(0.3).filled())
                    });
                // Draw the line on top of the area
                chart
                    .draw_series(LineSeries::new(points, color.stroke_width(2)))
                    .map_err(|e| anyhow!("Draw error: {e}"))?;
            }
            _ => unreachable!(),
        }
    }

    if args.series.len() > 1 {
        chart
            .configure_series_labels()
            .border_style(BLACK)
            .background_style(WHITE.mix(0.8))
            .draw()
            .map_err(|e| anyhow!("Legend draw error: {e}"))?;
    }

    Ok(())
}

fn draw_bar_chart<DB: DrawingBackend>(
    root: &DrawingArea<DB, plotters::coord::Shift>,
    args: &PlotArgs,
    title: &str,
    x_label: &str,
    y_label: &str,
) -> Result<()>
where
    DB::ErrorType: 'static,
{
    let first_series = &args.series[0];
    let n = first_series.y.len();
    let categories: Vec<String> = args
        .categories
        .clone()
        .unwrap_or_else(|| (0..n).map(|i| i.to_string()).collect());

    let y_min = args.y_range.map(|r| r[0]).unwrap_or_else(|| {
        args.series
            .iter()
            .flat_map(|s| s.y.iter())
            .copied()
            .fold(0.0_f64, f64::min)
            .min(0.0)
    });
    let y_max = args.y_range.map(|r| r[1]).unwrap_or_else(|| {
        let max = args
            .series
            .iter()
            .flat_map(|s| s.y.iter())
            .copied()
            .fold(f64::NEG_INFINITY, f64::max);
        max * 1.1
    });

    let mut builder = ChartBuilder::on(root);
    if !title.is_empty() {
        builder.caption(title, ("sans-serif", 24).into_font());
    }
    let mut chart = builder
        .margin(15)
        .x_label_area_size(40)
        .y_label_area_size(50)
        .build_cartesian_2d((0..n).into_segmented(), y_min..y_max)
        .map_err(|e| anyhow!("Chart build error: {e}"))?;

    let cats = categories.clone();
    let formatter = move |x: &SegmentValue<usize>| {
        if let SegmentValue::CenterOf(idx) = x {
            cats.get(*idx).cloned().unwrap_or_default()
        } else {
            String::new()
        }
    };
    let mut mesh = chart.configure_mesh();
    mesh.x_label_formatter(&formatter);
    if !x_label.is_empty() {
        mesh.x_desc(x_label);
    }
    if !y_label.is_empty() {
        mesh.y_desc(y_label);
    }
    mesh.draw().map_err(|e| anyhow!("Mesh draw error: {e}"))?;

    let num_series = args.series.len();
    for (si, s) in args.series.iter().enumerate() {
        let color = parse_color(s.color.as_deref(), si);
        let bar_width = 0.8 / num_series as f64;
        let offset = si as f64 * bar_width - 0.4 + bar_width / 2.0;
        let _ = offset; // grouped bars via offset not directly supported in segmented coord

        chart
            .draw_series((0..s.y.len()).map(|i| {
                let x0 = SegmentValue::CenterOf(i);
                Rectangle::new(
                    [(x0.clone(), 0.0), (x0, s.y[i])],
                    color
                        .mix(if num_series > 1 {
                            0.5 + 0.5 * (si as f64 / num_series as f64)
                        } else {
                            1.0
                        })
                        .filled(),
                )
            }))
            .map_err(|e| anyhow!("Draw error: {e}"))?
            .label(&s.name)
            .legend(move |(x, y)| Rectangle::new([(x, y - 5), (x + 20, y + 5)], color.filled()));
    }

    if args.series.len() > 1 {
        chart
            .configure_series_labels()
            .border_style(BLACK)
            .background_style(WHITE.mix(0.8))
            .draw()
            .map_err(|e| anyhow!("Legend draw error: {e}"))?;
    }

    Ok(())
}

fn draw_histogram<DB: DrawingBackend>(
    root: &DrawingArea<DB, plotters::coord::Shift>,
    args: &PlotArgs,
    title: &str,
    x_label: &str,
    y_label: &str,
) -> Result<()>
where
    DB::ErrorType: 'static,
{
    let first_series = &args.series[0];
    let values = &first_series.y;

    let data_min = values.iter().copied().fold(f64::INFINITY, f64::min);
    let data_max = values.iter().copied().fold(f64::NEG_INFINITY, f64::max);

    // Sturges' rule for bin count
    let num_bins = ((1.0 + 3.322 * (values.len() as f64).log10()).ceil() as usize).max(2);
    let bin_width = (data_max - data_min) / num_bins as f64;

    let mut bin_counts = vec![0u32; num_bins];
    for &v in values {
        let idx = ((v - data_min) / bin_width).floor() as usize;
        let idx = idx.min(num_bins - 1);
        bin_counts[idx] += 1;
    }

    let max_count = *bin_counts.iter().max().unwrap_or(&1);

    let x_range = args
        .x_range
        .map(|r| r[0]..r[1])
        .unwrap_or(data_min..data_max);
    let y_range = args
        .y_range
        .map(|r| r[0]..r[1])
        .unwrap_or(0.0..(max_count as f64 * 1.1));

    let mut builder = ChartBuilder::on(root);
    if !title.is_empty() {
        builder.caption(title, ("sans-serif", 24).into_font());
    }
    let mut chart = builder
        .margin(15)
        .x_label_area_size(40)
        .y_label_area_size(50)
        .build_cartesian_2d(x_range, y_range)
        .map_err(|e| anyhow!("Chart build error: {e}"))?;

    let mut mesh = chart.configure_mesh();
    if !x_label.is_empty() {
        mesh.x_desc(x_label);
    }
    let y_desc = if y_label.is_empty() { "Count" } else { y_label };
    mesh.y_desc(y_desc);
    mesh.draw().map_err(|e| anyhow!("Mesh draw error: {e}"))?;

    let color = parse_color(first_series.color.as_deref(), 0);
    chart
        .draw_series((0..num_bins).map(|i| {
            let x0 = data_min + i as f64 * bin_width;
            let x1 = x0 + bin_width;
            Rectangle::new([(x0, 0.0), (x1, bin_counts[i] as f64)], color.filled())
        }))
        .map_err(|e| anyhow!("Draw error: {e}"))?;

    Ok(())
}

// ---------------------------------------------------------------------------
// Helpers
// ---------------------------------------------------------------------------

fn compute_ranges(
    series: &[Series],
    x_range: Option<[f64; 2]>,
    y_range: Option<[f64; 2]>,
) -> (std::ops::Range<f64>, std::ops::Range<f64>) {
    let all_x: Vec<f64> = series
        .iter()
        .flat_map(|s| {
            s.x.clone()
                .unwrap_or_else(|| (0..s.y.len()).map(|i| i as f64).collect())
        })
        .collect();
    let all_y: Vec<f64> = series.iter().flat_map(|s| s.y.iter().copied()).collect();

    let x_min = all_x.iter().copied().fold(f64::INFINITY, f64::min);
    let x_max = all_x.iter().copied().fold(f64::NEG_INFINITY, f64::max);
    let y_min = all_y.iter().copied().fold(f64::INFINITY, f64::min);
    let y_max = all_y.iter().copied().fold(f64::NEG_INFINITY, f64::max);

    let pad = |min: f64, max: f64| -> (f64, f64) {
        if (max - min).abs() < f64::EPSILON {
            (min - 1.0, max + 1.0)
        } else {
            let margin = (max - min) * 0.05;
            (min - margin, max + margin)
        }
    };

    let xr = x_range.map(|r| r[0]..r[1]).unwrap_or_else(|| {
        let (lo, hi) = pad(x_min, x_max);
        lo..hi
    });
    let yr = y_range.map(|r| r[0]..r[1]).unwrap_or_else(|| {
        let (lo, hi) = pad(y_min, y_max);
        lo..hi
    });

    (xr, yr)
}

const PALETTE: &[RGBColor] = &[
    RGBColor(31, 119, 180),
    RGBColor(255, 127, 14),
    RGBColor(44, 160, 44),
    RGBColor(214, 39, 40),
    RGBColor(148, 103, 189),
    RGBColor(140, 86, 75),
    RGBColor(227, 119, 194),
    RGBColor(127, 127, 127),
];

fn parse_color(color_str: Option<&str>, index: usize) -> RGBColor {
    match color_str {
        Some(s) if s.starts_with('#') && s.len() == 7 => {
            let r = u8::from_str_radix(&s[1..3], 16).unwrap_or(0);
            let g = u8::from_str_radix(&s[3..5], 16).unwrap_or(0);
            let b = u8::from_str_radix(&s[5..7], 16).unwrap_or(0);
            RGBColor(r, g, b)
        }
        Some(name) => match name.to_lowercase().as_str() {
            "red" => RGBColor(214, 39, 40),
            "blue" => RGBColor(31, 119, 180),
            "green" => RGBColor(44, 160, 44),
            "orange" => RGBColor(255, 127, 14),
            "purple" => RGBColor(148, 103, 189),
            "brown" => RGBColor(140, 86, 75),
            "pink" => RGBColor(227, 119, 194),
            "gray" | "grey" => RGBColor(127, 127, 127),
            "black" => RGBColor(0, 0, 0),
            "yellow" => RGBColor(255, 215, 0),
            _ => PALETTE[index % PALETTE.len()],
        },
        None => PALETTE[index % PALETTE.len()],
    }
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;
    use tempfile::TempDir;

    use crate::tests::execute_tool;

    #[tokio::test]
    async fn test_line_chart() {
        let dir = TempDir::new().unwrap();
        let tool = PlotTool::new(dir.path().to_path_buf());

        let result = execute_tool(
            &tool,
            "call-1",
            json!({
                "chart_type": "line",
                "title": "Test Line",
                "series": [{"name": "A", "y": [1.0, 4.0, 2.0, 8.0, 5.0]}]
            }),
        )
        .await
        .unwrap();

        assert!(result.is_success());
        assert!(result.output().contains(".svg"));
        let files: Vec<_> = std::fs::read_dir(dir.path()).unwrap().collect();
        assert_eq!(files.len(), 1);
    }

    #[tokio::test]
    async fn test_bar_chart() {
        let dir = TempDir::new().unwrap();
        let tool = PlotTool::new(dir.path().to_path_buf());

        let result = execute_tool(
            &tool,
            "call-2",
            json!({
                "chart_type": "bar",
                "title": "Sales",
                "categories": ["Q1", "Q2", "Q3", "Q4"],
                "series": [{"name": "Revenue", "y": [100.0, 150.0, 120.0, 200.0]}]
            }),
        )
        .await
        .unwrap();

        assert!(result.is_success());
        assert!(result.output().contains(".svg"));
    }

    #[tokio::test]
    async fn test_scatter_chart_png() {
        let dir = TempDir::new().unwrap();
        let tool = PlotTool::new(dir.path().to_path_buf());

        let result = execute_tool(
            &tool,
            "call-3",
            json!({
                "chart_type": "scatter",
                "format": "png",
                "series": [{
                    "name": "Points",
                    "x": [1.0, 2.0, 3.0, 4.0, 5.0],
                    "y": [2.0, 3.5, 1.5, 4.0, 3.0],
                    "color": "#ff0000"
                }]
            }),
        )
        .await
        .unwrap();

        assert!(result.is_success());
        assert!(result.output().contains(".png"));
    }

    #[tokio::test]
    async fn test_histogram() {
        let dir = TempDir::new().unwrap();
        let tool = PlotTool::new(dir.path().to_path_buf());

        let result = execute_tool(
            &tool,
            "call-4",
            json!({
                "chart_type": "histogram",
                "title": "Distribution",
                "series": [{"name": "Values", "y": [1.0, 1.5, 2.0, 2.5, 3.0, 3.5, 4.0, 2.0, 2.5, 3.0]}]
            }),
        )
        .await
        .unwrap();

        assert!(result.is_success());
    }

    #[tokio::test]
    async fn test_area_chart() {
        let dir = TempDir::new().unwrap();
        let tool = PlotTool::new(dir.path().to_path_buf());

        let result = execute_tool(
            &tool,
            "call-5",
            json!({
                "chart_type": "area",
                "series": [
                    {"name": "Series A", "y": [1.0, 3.0, 2.0, 5.0]},
                    {"name": "Series B", "y": [2.0, 1.0, 4.0, 3.0]}
                ]
            }),
        )
        .await
        .unwrap();

        assert!(result.is_success());
    }

    #[tokio::test]
    async fn test_empty_series_fails() {
        let dir = TempDir::new().unwrap();
        let tool = PlotTool::new(dir.path().to_path_buf());

        let result = execute_tool(
            &tool,
            "call-6",
            json!({
                "chart_type": "line",
                "series": []
            }),
        )
        .await
        .unwrap();

        assert!(!result.is_success());
    }
}
