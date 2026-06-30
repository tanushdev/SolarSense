import { useMetrics } from "@/hooks/useApi"
import { Card, CardContent } from "@/components/ui/card"

function MetricRow({ label, value }: { label: string; value: string }) {
  return (
    <div className="flex justify-between items-center py-3 px-5 border-b border-bug-hairline last:border-b-0">
      <span className="mono-sm text-bug-muted">{label}</span>
      <span className="text-[15px] font-mono tracking-tight text-bug-text">{value}</span>
    </div>
  )
}

function SectionHeader({ label }: { label: string }) {
  return (
    <div className="px-5 py-2.5 border-b border-bug-hairline">
      <span className="mono-label text-bug-muted-soft">{label}</span>
    </div>
  )
}

export default function ModelPerformancePage() {
  const { data: metrics, isLoading } = useMetrics()

  return (
    <div className="space-y-8 animate-slide-up">
      <div className="border-b border-bug-hairline pb-4">
        <h1 className="display-lg text-bug-text">MISSION TELEMETRY</h1>
        <p className="mono-sm text-bug-muted mt-1">
          {isLoading ? "ACQUIRING..." : metrics ? `MODEL: ${metrics.model} · ${metrics.total_predictions?.toLocaleString() ?? "—"} PREDICTIONS` : "NO MODEL LOADED"}
        </p>
      </div>

      {isLoading ? (
        <div className="flex items-center justify-center py-24">
          <span className="mono-label text-bug-muted">ACQUIRING TELEMETRY...</span>
        </div>
      ) : !metrics ? (
        <div className="flex items-center justify-center py-24">
          <span className="mono-label text-bug-muted">NO TELEMETRY DATA</span>
        </div>
      ) : (
        <div className="grid grid-cols-1 lg:grid-cols-2 gap-6">
          {/* Performance Gauges — simple text display */}
          <Card>
            <SectionHeader label="SKILL SCORES" />
            <MetricRow label="TSS (True Skill Statistic)" value={metrics.tss.toFixed(4)} />
            <MetricRow label="HSS (Heidke Skill Score)" value={metrics.hss.toFixed(4)} />
            <MetricRow label="ROC-AUC" value={metrics.roc_auc.toFixed(4)} />
            <MetricRow label="F1 Score" value={metrics.f1.toFixed(4)} />
          </Card>

          <Card>
            <SectionHeader label="ERROR METRICS" />
            <MetricRow label="Brier Score" value={metrics.brier.toFixed(4)} />
            <MetricRow label="ECE" value={metrics.ece.toFixed(4)} />
            <MetricRow label="Precision" value={metrics.precision.toFixed(4)} />
            <MetricRow label="Recall" value={metrics.recall.toFixed(4)} />
            <MetricRow label="False Alarm Rate" value={(metrics.false_alarm_rate * 100).toFixed(2) + "%"} />
          </Card>

          <Card>
            <SectionHeader label="TEMPORAL METRICS" />
            <MetricRow label="Avg Lead Time" value={metrics.avg_lead_time.toFixed(1) + " min"} />
            <MetricRow label="Accuracy" value={(metrics.accuracy * 100).toFixed(2) + "%"} />
            {metrics.prediction_accuracy != null && (
              <MetricRow label="Prediction Accuracy" value={(metrics.prediction_accuracy * 100).toFixed(2) + "%"} />
            )}
          </Card>

          <Card>
            <SectionHeader label="PREDICTION COUNTS" />
            <MetricRow label="Total Predictions" value={metrics.total_predictions?.toLocaleString() ?? "—"} />
            <MetricRow label="Correct Predictions" value={metrics.correct_predictions?.toLocaleString() ?? "—"} />
          </Card>
        </div>
      )}
    </div>
  )
}
