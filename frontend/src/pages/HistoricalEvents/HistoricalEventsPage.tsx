import { useState, useMemo } from "react"
import { Search, ChevronLeft, ChevronRight } from "lucide-react"
import { useHistory } from "@/hooks/useApi"
import { formatDate, cn, classColor } from "@/lib/utils"
import { Card } from "@/components/ui/card"
import type { NowcastEvent } from "@/types"

const PAGE_SIZES = [25, 50, 100]

export default function HistoricalEventsPage() {
  const { data: events = [], isLoading } = useHistory()
  const [search, setSearch] = useState("")
  const [classFilter, setClassFilter] = useState<string>("all")
  const [page, setPage] = useState(0)
  const [pageSize, setPageSize] = useState(25)

  const filtered = useMemo(() => {
    return (events as NowcastEvent[]).filter((e) => {
      const matchSearch = !search || e.flare_class.toLowerCase().includes(search.toLowerCase())
      const matchClass = classFilter === "all" || e.flare_class === classFilter
      return matchSearch && matchClass
    })
  }, [events, search, classFilter])

  const totalPages = Math.max(1, Math.ceil(filtered.length / pageSize))
  const safePage = Math.min(page, totalPages - 1)
  const paged = filtered.slice(safePage * pageSize, (safePage + 1) * pageSize)

  return (
    <div className="space-y-8 animate-slide-up">
      <div className="border-b border-bug-hairline pb-4">
        <h1 className="display-lg text-bug-text">HISTORICAL EVENTS</h1>
        <p className="mono-sm text-bug-muted mt-1">{events.length} detected flare events</p>
      </div>

      <div className="flex flex-wrap items-center gap-3">
        <div className="relative flex-1 max-w-xs">
          <Search className="absolute left-3 top-1/2 -translate-y-1/2 h-3.5 w-3.5 text-bug-muted-soft" />
          <input type="text" placeholder="Search events..." value={search}
            onChange={(e) => { setSearch(e.target.value); setPage(0) }}
            className="w-full bg-transparent border border-bug-hairline pl-9 pr-3 py-2 text-[13px] text-bug-text placeholder:text-bug-muted-soft focus:outline-none font-mono" />
        </div>
        <div className="flex gap-1">
          {["all", "X", "M", "C", "B", "A"].map((cls) => (
            <button key={cls} onClick={() => { setClassFilter(cls); setPage(0) }}
              className={cn("px-3 py-1.5 mono-sm border transition-colors",
                classFilter === cls ? "border-bug-text text-bug-text" : "border-bug-hairline text-bug-muted hover:border-bug-muted"
              )}>{cls === "all" ? "ALL" : cls}</button>
          ))}
        </div>
      </div>

      <Card>
        <div className="overflow-x-auto">
          <table className="w-full text-[13px] font-mono">
            <thead>
              <tr className="border-b border-bug-hairline text-bug-muted">
                <th className="text-left p-3 font-normal mono-sm">TIME</th>
                <th className="text-left p-3 font-normal mono-sm">CLASS</th>
                <th className="text-right p-3 font-normal mono-sm">PEAK FLUX</th>
                <th className="text-right p-3 font-normal mono-sm">PREDICTION</th>
                <th className="text-center p-3 font-normal mono-sm">QUALITY</th>
              </tr>
            </thead>
            <tbody>
              {isLoading ? (
                <tr><td colSpan={5} className="text-center p-8 mono-sm text-bug-muted">loading events...</td></tr>
              ) : paged.length === 0 ? (
                <tr><td colSpan={5} className="text-center p-8 mono-sm text-bug-muted">no events found</td></tr>
              ) : paged.map((event, idx) => (
                <tr key={idx} className="border-b border-bug-hairline last:border-b-0 hover:bg-bug-surface transition-colors">
                  <td className="p-3 text-bug-body">{formatDate(event.peak_time)}</td>
                  <td className="p-3">
                    <span className={classColor(event.flare_class)}>
                      {event.flare_class}{event.flare_subclass > 0 ? event.flare_subclass.toFixed(1) : ""}
                    </span>
                  </td>
                  <td className="p-3 text-right text-bug-body">{event.peak_hard_flux.toFixed(1)}</td>
                  <td className="p-3 text-right mono-sm text-bug-muted">{event.confirmation}</td>
                  <td className="p-3 text-center mono-sm">
                    <span className={event.quality === 0 ? "text-bug-success" : event.quality === 1 ? "text-bug-warning" : "text-bug-muted-soft"}>
                      {event.quality === 0 ? "HIGH" : event.quality === 1 ? "MED" : "LOW"}
                    </span>
                  </td>
                </tr>
              ))}
            </tbody>
          </table>
        </div>

        {/* Pagination */}
        <div className="flex items-center justify-between px-5 py-3 border-t border-bug-hairline mono-sm">
          <div className="flex items-center gap-2 text-bug-muted">
            <span>ROWS</span>
            <select value={pageSize} onChange={e => { setPageSize(Number(e.target.value)); setPage(0) }}
              className="bg-transparent border border-bug-hairline px-2 py-1 text-bug-text font-mono mono-sm focus:outline-none">
              {PAGE_SIZES.map(s => <option key={s} value={s}>{s}</option>)}
            </select>
            <span>{filtered.length} TOTAL</span>
          </div>

          <div className="flex items-center gap-2">
            <button onClick={() => setPage(Math.max(0, safePage - 1))}
              disabled={safePage === 0}
              className="p-1 border border-bug-hairline disabled:opacity-30 hover:border-bug-muted transition-colors">
              <ChevronLeft className="h-3.5 w-3.5" />
            </button>
            <span className="text-bug-text">
              {safePage + 1} / {totalPages}
            </span>
            <button onClick={() => setPage(Math.min(totalPages - 1, safePage + 1))}
              disabled={safePage >= totalPages - 1}
              className="p-1 border border-bug-hairline disabled:opacity-30 hover:border-bug-muted transition-colors">
              <ChevronRight className="h-3.5 w-3.5" />
            </button>
          </div>
        </div>
      </Card>
    </div>
  )
}
