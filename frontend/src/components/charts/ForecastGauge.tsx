import { cn } from "@/lib/utils"

export function ForecastGauge({ probability, predictedClass }: { probability: number; predictedClass: string }) {
  const pct = Math.round(probability * 100)
  return (
    <div className="flex flex-col items-center">
      <span className="text-[56px] font-light tracking-tight text-bug-text font-mono">{pct}%</span>
      <span className="mono-label text-bug-muted mt-1">PROBABILITY</span>
      <span className="text-[28px] font-light tracking-tight text-bug-text font-mono mt-2">{predictedClass}</span>
    </div>
  )
}
