import { useDatasets } from "@/hooks/useApi"
import { Card, CardContent } from "@/components/ui/card"
import { cn } from "@/lib/utils"

function DatasetRow({ name, info }: { name: string; info: { exists: boolean; files: number; size_mb: number; path: string } }) {
  return (
    <div className={cn(
      "flex items-center justify-between py-4 px-5 border-b border-bug-hairline last:border-b-0 transition-colors",
      info.exists ? "" : "opacity-40"
    )}>
      <div className="flex items-center gap-3">
        <span className={cn("mono-sm", info.exists ? "text-bug-success" : "text-bug-muted-soft")}>
          {info.exists ? "●" : "○"}
        </span>
        <div>
          <div className="text-[15px] font-light tracking-tight text-bug-text uppercase">{name}</div>
          <div className="mono-sm text-bug-muted-soft mt-0.5">{info.path}</div>
        </div>
      </div>
      <div className="flex items-center gap-6 mono-sm text-bug-muted">
        <span>{info.exists ? `${info.files.toLocaleString()} FILES` : "OFFLINE"}</span>
        <span className="text-right w-20">{info.exists ? `${info.size_mb.toFixed(1)} MB` : "—"}</span>
      </div>
    </div>
  )
}

export default function DatasetsPage() {
  const { data: datasets, isLoading } = useDatasets()

  return (
    <div className="space-y-8 animate-slide-up">
      <div className="border-b border-bug-hairline pb-4">
        <h1 className="display-lg text-bug-text">DATA ARCHIVE</h1>
        <p className="mono-sm text-bug-muted mt-1">
          {isLoading ? "SCANNING FILESYSTEM..." : datasets ? `${Object.values(datasets).filter(d => d.exists).length}/5 NODES ONLINE` : "BACKEND OFFLINE"}
        </p>
      </div>

      {isLoading ? (
        <div className="flex items-center justify-center py-24">
          <span className="mono-label text-bug-muted">SCANNING...</span>
        </div>
      ) : !datasets ? (
        <div className="flex items-center justify-center py-24">
          <span className="mono-label text-bug-muted">NO DATA ARCHIVE ACCESS</span>
        </div>
      ) : (
        <Card>
          <CardContent className="p-0 divide-y divide-bug-hairline">
            {Object.entries(datasets).map(([name, info]) => (
              <DatasetRow key={name} name={name} info={info} />
            ))}
          </CardContent>
        </Card>
      )}
    </div>
  )
}
