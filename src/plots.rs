//! Interactive charts built with Plotly, plus a single-page HTML dashboard
//! that collects them.
//!
//! Colors follow a validated, colour-blind-safe categorical palette. Each
//! entity keeps its colour across every chart (sigmoid is always blue, and
//! so on), and train vs. validation is encoded by line style as well as
//! colour, so no chart relies on colour alone.

use std::fmt::Write as _;
use std::fs;
use std::io;
use std::path::Path;

use plotly::common::{
    ColorScale, ColorScaleElement, DashType, ErrorData, ErrorType, Font, Line, Marker, Mode, Title,
};
use plotly::layout::{Annotation, Axis, HoverMode, Layout, Legend, Margin};
use plotly::{HeatMap, Plot, Scatter};

use crate::activation::Activation;
use crate::metrics::ConfusionMatrix;

/// Categorical palette, slots in fixed order.
pub const PALETTE: [&str; 4] = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100"];

/// Sequential blue ramp (light → dark) for heatmaps.
const SEQUENTIAL: [&str; 5] = ["#cde2fb", "#86b6ef", "#3987e5", "#1c5cab", "#0d366b"];

const SURFACE: &str = "#fcfcfb";
const TEXT_PRIMARY: &str = "#0b0b0b";
const TEXT_SECONDARY: &str = "#52514e";
const GRID: &str = "#e8e7e3";
const FONT_FAMILY: &str = "Inter, system-ui, -apple-system, Segoe UI, sans-serif";

/// Fixed colour per activation, so it is identical in every chart.
#[must_use]
pub fn activation_color(activation: Activation) -> &'static str {
    match activation {
        Activation::Sigmoid => PALETTE[0],
        Activation::Tanh => PALETTE[1],
        Activation::Relu => PALETTE[2],
    }
}

/// One train/validation pair of learning curves.
#[derive(Debug, Clone)]
pub struct CurvePair<'a> {
    /// Legend label, e.g. `"lr = 0.05"`.
    pub label: String,
    /// Line colour (a palette constant).
    pub color: &'static str,
    /// Training values per epoch.
    pub train: &'a [f64],
    /// Validation values per epoch.
    pub validation: &'a [f64],
}

fn axis(title: &str) -> Axis {
    Axis::new()
        .title(Title::with_text(title).font(Font::new().color(TEXT_SECONDARY).size(13)))
        .grid_color(GRID)
        .zero_line(false)
        .line_color(GRID)
        .tick_font(Font::new().color(TEXT_SECONDARY).size(12))
}

fn base_layout(title: &str, x_title: &str, y_title: &str) -> Layout {
    Layout::new()
        .title(Title::with_text(title).font(Font::new().size(17).color(TEXT_PRIMARY)))
        .font(Font::new().family(FONT_FAMILY).color(TEXT_PRIMARY))
        .paper_background_color(SURFACE)
        .plot_background_color(SURFACE)
        .x_axis(axis(x_title))
        .y_axis(axis(y_title))
        .legend(
            Legend::new()
                .orientation(plotly::common::Orientation::Horizontal)
                .y(-0.2)
                .font(Font::new().color(TEXT_SECONDARY)),
        )
        .margin(Margin::new().top(60).bottom(80).left(70).right(30))
        .height(460)
}

/// Train (dashed) and validation (solid) curves for several configurations.
#[must_use]
pub fn learning_curves(title: &str, y_title: &str, pairs: &[CurvePair<'_>]) -> Plot {
    let mut plot = Plot::new();
    for pair in pairs {
        let epochs: Vec<usize> = (0..pair.validation.len()).collect();
        plot.add_trace(
            Scatter::new(epochs.clone(), pair.validation.to_vec())
                .name(format!("{} · validation", pair.label))
                .legend_group(pair.label.clone())
                .mode(Mode::Lines)
                .line(Line::new().color(pair.color).width(2.0))
                .hover_template(format!("{} val: %{{y:.4f}}<extra></extra>", pair.label)),
        );
        plot.add_trace(
            Scatter::new(epochs, pair.train.to_vec())
                .name(format!("{} · train", pair.label))
                .legend_group(pair.label.clone())
                .mode(Mode::Lines)
                .line(
                    Line::new()
                        .color(pair.color)
                        .width(2.0)
                        .dash(DashType::Dash),
                )
                .hover_template(format!("{} train: %{{y:.4f}}<extra></extra>", pair.label)),
        );
    }
    plot.set_layout(base_layout(title, "Epoch", y_title).hover_mode(HoverMode::XUnified));
    plot
}

/// A line with one `(mean, std)` value per x position.
#[derive(Debug, Clone)]
pub struct ErrorBarSeries {
    /// Legend label.
    pub name: String,
    /// Line colour (a palette constant).
    pub color: &'static str,
    /// `(mean, std)` for each x label.
    pub values: Vec<(f64, f64)>,
}

/// One line per series over categorical x values, with ±1 std error bars.
#[must_use]
pub fn metric_with_error_bars(
    title: &str,
    x_title: &str,
    y_title: &str,
    x_labels: &[String],
    series: &[ErrorBarSeries],
) -> Plot {
    let mut plot = Plot::new();
    for ErrorBarSeries {
        name,
        color,
        values,
    } in series
    {
        let means: Vec<f64> = values.iter().map(|v| v.0).collect();
        let stds: Vec<f64> = values.iter().map(|v| v.1).collect();
        plot.add_trace(
            Scatter::new(x_labels.to_vec(), means)
                .name(name.clone())
                .mode(Mode::LinesMarkers)
                .line(Line::new().color(*color).width(2.0))
                .marker(Marker::new().color(*color).size(9))
                .error_y(
                    ErrorData::new(ErrorType::Data)
                        .array(stds)
                        .color(*color)
                        .thickness(1.5)
                        .width(4),
                )
                .hover_template(format!("{name}: %{{y:.4f}}<extra></extra>")),
        );
    }
    let mut layout = base_layout(title, x_title, y_title).hover_mode(HoverMode::XUnified);
    layout = layout.x_axis(axis(x_title).type_(plotly::layout::AxisType::Category));
    plot.set_layout(layout);
    plot
}

/// Inputs for [`heatmap`]. `z[row][col]` is drawn at `(x_labels[col], y_labels[row])`.
#[derive(Debug, Clone)]
pub struct HeatmapSpec<'a> {
    /// Chart title.
    pub title: &'a str,
    /// X axis title.
    pub x_title: &'a str,
    /// Y axis title.
    pub y_title: &'a str,
    /// Column labels.
    pub x_labels: &'a [String],
    /// Row labels.
    pub y_labels: &'a [String],
    /// Cell values, one inner vector per row.
    pub z: &'a [Vec<f64>],
    /// `true` when *lower* is better, so low values get the darkest colour.
    pub reverse: bool,
}

/// A heatmap with the value printed in every cell.
#[must_use]
pub fn heatmap(spec: &HeatmapSpec<'_>, value_format: impl Fn(f64) -> String) -> Plot {
    let HeatmapSpec {
        title,
        x_title,
        y_title,
        x_labels,
        y_labels,
        z,
        reverse,
    } = *spec;
    let n = SEQUENTIAL.len() - 1;
    let scale: Vec<ColorScaleElement> = SEQUENTIAL
        .iter()
        .enumerate()
        .map(|(i, c)| ColorScaleElement(i as f64 / n as f64, (*c).to_string()))
        .collect();

    let finite = z.iter().flatten().copied().filter(|v| v.is_finite());
    let (lo, hi) = finite.fold((f64::INFINITY, f64::NEG_INFINITY), |(a, b), v| {
        (a.min(v), b.max(v))
    });
    let mid = f64::midpoint(lo, hi);

    let mut annotations = Vec::new();
    // On a category axis, position k is the k-th label; using indices avoids
    // Plotly re-interpreting numeric-looking labels such as "16" as numbers.
    for (row_i, row) in z.iter().enumerate() {
        for (col_i, &v) in row.iter().enumerate() {
            // Dark text on light cells, light text on dark cells.
            let dark_cell = if reverse { v < mid } else { v > mid };
            annotations.push(
                Annotation::new()
                    .x(col_i as f64)
                    .y(row_i as f64)
                    .text(value_format(v))
                    .show_arrow(false)
                    .font(Font::new().size(12).color(if dark_cell {
                        "#ffffff"
                    } else {
                        TEXT_PRIMARY
                    })),
            );
        }
    }

    let mut plot = Plot::new();
    plot.add_trace(
        HeatMap::new(x_labels.to_vec(), y_labels.to_vec(), z.to_vec())
            .color_scale(ColorScale::Vector(scale))
            .reverse_scale(reverse)
            .x_gap(2)
            .y_gap(2)
            .hover_template(format!(
                "{x_title} %{{x}}<br>{y_title} %{{y}}<br>%{{z:.4f}}<extra></extra>"
            )),
    );
    plot.set_layout(
        base_layout(title, x_title, y_title)
            .x_axis(axis(x_title).type_(plotly::layout::AxisType::Category))
            .y_axis(axis(y_title).type_(plotly::layout::AxisType::Category))
            .annotations(annotations)
            // Extra left margin so long row labels do not collide with the axis title.
            .margin(Margin::new().top(60).bottom(80).left(120).right(30))
            .height(420),
    );
    plot
}

/// Confusion matrix as an annotated heatmap (rows = actual, columns = predicted).
#[must_use]
pub fn confusion_matrix(title: &str, cm: &ConfusionMatrix, class_names: &[String]) -> Plot {
    let short: Vec<String> = class_names
        .iter()
        .map(|n| n.trim_start_matches("Iris-").to_string())
        .collect();
    // Plotly draws the first row at the bottom; reverse so row 0 is on top.
    let rows: Vec<Vec<f64>> = cm
        .as_rows()
        .iter()
        .rev()
        .map(|r| r.iter().map(|&c| c as f64).collect())
        .collect();
    let y_labels: Vec<String> = short.iter().rev().cloned().collect();
    heatmap(
        &HeatmapSpec {
            title,
            x_title: "Predicted class",
            y_title: "Actual class",
            x_labels: &short,
            y_labels: &y_labels,
            z: &rows,
            reverse: false,
        },
        |v| format!("{v:.0}"),
    )
}

/// A single-page HTML report: a title, then a sequence of sections, each
/// holding a chart or an HTML snippet (e.g. a table).
#[derive(Debug, Default)]
pub struct Dashboard {
    title: String,
    subtitle: String,
    sections: Vec<(String, String, String)>,
}

impl Dashboard {
    /// Starts a dashboard.
    #[must_use]
    pub fn new(title: impl Into<String>, subtitle: impl Into<String>) -> Self {
        Self {
            title: title.into(),
            subtitle: subtitle.into(),
            sections: Vec::new(),
        }
    }

    /// Adds a chart with a heading and an explanatory caption.
    pub fn add_plot(&mut self, heading: &str, caption: &str, plot: &Plot) {
        let id = format!("plot-{}", self.sections.len());
        self.sections.push((
            heading.to_string(),
            caption.to_string(),
            plot.to_inline_html(Some(&id)),
        ));
    }

    /// Adds raw HTML (already escaped) with a heading and caption.
    pub fn add_html(&mut self, heading: &str, caption: &str, html: String) {
        self.sections
            .push((heading.to_string(), caption.to_string(), html));
    }

    /// Writes the dashboard as a standalone HTML file.
    ///
    /// # Errors
    ///
    /// Returns any I/O error from writing the file.
    pub fn write(&self, path: impl AsRef<Path>) -> io::Result<()> {
        let mut body = String::new();
        for (heading, caption, content) in &self.sections {
            let _ = write!(
                body,
                r#"<section class="card"><h2>{}</h2><p class="caption">{}</p>{}</section>"#,
                escape(heading),
                escape(caption),
                content
            );
        }
        let html = format!(
            r#"<!doctype html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>{title}</title>
<script src="https://cdn.plot.ly/plotly-3.0.1.min.js"></script>
<style>
  :root {{ --surface:{SURFACE}; --page:#f4f3f0; --text:{TEXT_PRIMARY}; --muted:{TEXT_SECONDARY}; --grid:{GRID}; }}
  * {{ box-sizing: border-box; }}
  body {{ margin:0; background:var(--page); color:var(--text); font-family:{FONT_FAMILY}; }}
  header {{ max-width:1100px; margin:0 auto; padding:40px 24px 8px; }}
  h1 {{ font-size:28px; margin:0 0 6px; letter-spacing:-0.01em; }}
  header p {{ color:var(--muted); margin:0; }}
  main {{ max-width:1100px; margin:0 auto; padding:16px 24px 48px; display:grid; gap:20px; }}
  .card {{ background:var(--surface); border:1px solid var(--grid); border-radius:12px; padding:20px 24px; }}
  h2 {{ font-size:18px; margin:0 0 4px; }}
  .caption {{ color:var(--muted); margin:0 0 12px; font-size:14px; line-height:1.5; }}
  table {{ border-collapse:collapse; width:100%; font-size:14px; font-variant-numeric:tabular-nums; }}
  th, td {{ padding:8px 10px; border-bottom:1px solid var(--grid); text-align:right; }}
  th:first-child, td:first-child {{ text-align:left; }}
  th {{ color:var(--muted); font-weight:600; }}
  tr.best td {{ font-weight:700; }}
</style>
</head>
<body>
<header><h1>{title}</h1><p>{subtitle}</p></header>
<main>{body}</main>
</body>
</html>
"#,
            title = escape(&self.title),
            subtitle = escape(&self.subtitle),
        );
        fs::write(path, html)
    }
}

/// Minimal HTML escaping for text content.
#[must_use]
pub fn escape(s: &str) -> String {
    s.replace('&', "&amp;")
        .replace('<', "&lt;")
        .replace('>', "&gt;")
        .replace('"', "&quot;")
}

/// Renders rows of cells as an HTML table. Rows listed in `highlight` are bold.
#[must_use]
pub fn html_table(header: &[&str], rows: &[Vec<String>], highlight: &[usize]) -> String {
    let mut html = String::from("<table><thead><tr>");
    for h in header {
        let _ = write!(html, "<th>{}</th>", escape(h));
    }
    html.push_str("</tr></thead><tbody>");
    for (i, row) in rows.iter().enumerate() {
        let class = if highlight.contains(&i) {
            r#" class="best""#
        } else {
            ""
        };
        let _ = write!(html, "<tr{class}>");
        for cell in row {
            let _ = write!(html, "<td>{}</td>", escape(cell));
        }
        html.push_str("</tr>");
    }
    html.push_str("</tbody></table>");
    html
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn escape_neutralizes_html() {
        assert_eq!(escape("<a & \"b\">"), "&lt;a &amp; &quot;b&quot;&gt;");
    }

    #[test]
    fn table_highlights_requested_rows() {
        let html = html_table(&["a"], &[vec!["1".into()], vec!["2".into()]], &[1]);
        assert_eq!(html.matches(r#"class="best""#).count(), 1);
    }

    #[test]
    fn charts_serialize_their_data() {
        let train = [1.0, 0.5];
        let val = [1.1, 0.6];
        let plot = learning_curves(
            "t",
            "loss",
            &[CurvePair {
                label: "x".into(),
                color: PALETTE[0],
                train: &train,
                validation: &val,
            }],
        );
        let json = plot.to_json();
        assert!(json.contains("0.6") && json.contains("x · train"));
        let cm = ConfusionMatrix::new(&[0, 1], &[0, 1], 2);
        let names = vec!["Iris-a".to_string(), "Iris-b".to_string()];
        assert!(
            confusion_matrix("cm", &cm, &names)
                .to_json()
                .contains("Predicted class")
        );
    }
}
