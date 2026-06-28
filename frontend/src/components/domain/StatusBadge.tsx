import { cn } from "@/lib/utils"

export function StatusBadge({ status, label, className }: { status: string; label?: string; className?: string }) {
  return (
    <span className={cn("mono-sm text-bug-muted", className)}>
      {label ?? status.toUpperCase()}
    </span>
  )
}
