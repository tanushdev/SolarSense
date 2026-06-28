import { useState } from "react"
import { useValidation } from "@/hooks/useApi"
import { Card, CardContent } from "@/components/ui/card"
import { cn, classColor } from "@/lib/utils"
import type { ValidationMatch } from "@/types"

function StatCell({ label, value, color }: { label: string; value: string; color?: string }) {
  return (
    <div className="text-center border border-bug-hairline p-4">
      <div className="mono-sm text-bug-muted mb-1">{label}</div>
      <div className={cn("text-[22px] font-light tracking-tight font-mono", color ?? "text-bug-text")}>{value}</div>
    </div>
  )
}

export default function ValidationPage() {
  const [startDate, setStartDate] = useState("2024-07-01")
  const [endDate, setEndDate] = useState("2024-12-31")
  const { data: report, isLoading } = useValidation(startDate, endDate)

  return (
    <div className="space-y-8 animate-slide-up">
      <div className="border-b border-bug-hairline pb-4">
        <div className="flex items-center justify-between">
          <div>
            <h1 className="display-lg text-bug-text">VERIFICATION</h1>
            <p className="mono-sm text-bug-muted mt-1">
              {isLoading ? "SCANNING..." : report ? `${report.matched} MATCHES · ${report.goes_events} GOES EVENTS` : "AWAITING SIGNAL"}
            </p>
          </div>
          <div className="flex items-center gap-3 mono-sm text-bug-muted">
            <input type="date" value={startDate} onChange={e => setStartDate(e.target.value)}
              className="bg-transparent border border-bug-hairline px-2.5 py-1.5 text-bug-text font-mono mono-sm" />
            <span>→</span>
            <input type="date" value={endDate} onChange={e => setEndDate(e.target.value)}
              className="bg-transparent border border-bug-hairline px-2.5 py-1.5 text-bug-text font-mono mono-sm" />
          </div>
        </div>
      </div>

      {isLoading ? (
        <div className="flex items-center justify-center py-24">
          <span className="mono-label text-bug-muted">SCANNING FOR MATCHES...</span>
        </div>
      ) : !report ? (
        <div className="flex items-center justify-center py-24">
          <span className="mono-label text-bug-muted">NO VERIFICATION DATA</span>
        </div>
      ) : (
        <>
          <div className="grid grid-cols-2 md:grid-cols-4 gap-3">
            <StatCell label="DETECTION RATE" value={`${(report.detection_rate * 100).toFixed(1)}%`} color={report.detection_rate > 0.5 ? "text-bug-success" : "text-bug-warning"} />
            <StatCell label="PRECISION" value={`${(report.precision * 100).toFixed(1)}%`} color={report.precision > 0.5 ? "text-bug-success" : "text-bug-warning"} />
            <StatCell label="RECALL" value={`${(report.recall * 100).toFixed(1)}%`} color={report.recall > 0.5 ? "text-bug-success" : "text-bug-warning"} />
            <StatCell label="MATCHED" value={`${report.matched}`} color="text-bug-link" />
          </div>

          <div className="grid grid-cols-2 md:grid-cols-4 gap-3">
            <StatCell label="OUR EVENTS" value={`${report.our_events}`} />
            <StatCell label="GOES EVENTS" value={`${report.goes_events}`} />
            <StatCell label="MISSED" value={`${report.missed}`} color="text-bug-warning" />
            <StatCell label="FALSE ALARMS" value={`${report.false_alarms}`} color="text-bug-muted-soft" />
          </div>

          {report.matches.length > 0 && (
            <Card>
              <div className="px-5 py-2.5 border-b border-bug-hairline">
                <span className="mono-label text-bug-muted-soft">MATCHED EVENTS · {report.matches.length} ENTRIES</span>
              </div>
              <div className="overflow-x-auto">
                <table className="w-full text-[13px] font-mono">
                  <thead>
                    <tr className="border-b border-bug-hairline text-bug-muted">
                      <th className="text-left p-3 font-normal mono-sm">OUR PEAK</th>
                      <th className="text-left p-3 font-normal mono-sm">CLASS</th>
                      <th className="text-left p-3 font-normal mono-sm">GOES PEAK</th>
                      <th className="text-left p-3 font-normal mono-sm">GOES</th>
                      <th className="text-right p-3 font-normal mono-sm">Δ MIN</th>
                    </tr>
                  </thead>
                  <tbody>
                    {report.matches.map((m: ValidationMatch, i: number) => (
                      <tr key={i} className="border-b border-bug-hairline last:border-b-0 hover:bg-bug-surface transition-colors">
                        <td className="p-3 text-bug-body">{new Date(m.our_peak).toLocaleString("en-IN", { timeZone: "UTC", hour12: false })}</td>
                        <td className="p-3"><span className={classColor(m.our_class)}>{m.our_class}</span></td>
                        <td className="p-3 text-bug-body">{new Date(m.goes_peak).toLocaleString("en-IN", { timeZone: "UTC", hour12: false })}</td>
                        <td className="p-3"><span className={classColor(m.goes_class)}>{m.goes_class}</span></td>
                        <td className="p-3 text-right text-bug-text">{Math.abs(m.delta_min).toFixed(1)}</td>
                      </tr>
                    ))}
                  </tbody>
                </table>
              </div>
            </Card>
          )}
        </>
      )}
    </div>
  )
}
