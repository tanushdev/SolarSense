import { cn } from "@/lib/utils"

const variants = {
  default: "text-bug-muted",
  success: "text-bug-success",
  warning: "text-bug-warning",
  danger: "text-bug-text",
  info: "text-bug-link",
}

export function Badge({ variant = "default", className, children, ...props }: { variant?: keyof typeof variants } & React.HTMLAttributes<HTMLSpanElement>) {
  return (
    <span className={cn("mono-label inline", variants[variant], className)} {...props}>
      {children}
    </span>
  )
}
