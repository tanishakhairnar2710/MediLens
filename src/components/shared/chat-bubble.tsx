"use client";

import { motion } from "framer-motion";
import { Activity } from "lucide-react";
import ReactMarkdown from "react-markdown";
import remarkGfm from "remark-gfm";
import { Avatar, AvatarFallback } from "@/components/ui/avatar";
import type { ChatMessage } from "@/lib/types";
import { cn } from "@/lib/utils";

export function ChatBubble({
  message,
}: {
  message: ChatMessage;
}) {
  const isUser = message.role === "user";

  return (
    <motion.div
      initial={{ opacity: 0, y: 10 }}
      animate={{ opacity: 1, y: 0 }}
      transition={{ duration: 0.3 }}
      className={cn(
        "flex items-end gap-2.5",
        isUser && "flex-row-reverse"
      )}
    >
      {/* Avatar */}
      {isUser ? (
        <Avatar className="h-7 w-7 shrink-0">
          <AvatarFallback className="text-[11px]">
            AM
          </AvatarFallback>
        </Avatar>
      ) : (
        <span className="flex h-7 w-7 shrink-0 items-center justify-center rounded-full bg-primary text-white">
          <Activity className="h-3.5 w-3.5" />
        </span>
      )}

      {/* Message */}
      <div className="max-w-[88%] sm:max-w-[78%]">
        <div
          className={cn(
            "rounded-2xl px-4 py-3 text-sm leading-relaxed",
            isUser
              ? "rounded-br-md bg-primary text-white"
              : "rounded-bl-md border border-border bg-white text-foreground"
          )}
        >
          {isUser ? (
            /* User messages */
            <div className="whitespace-pre-wrap break-words">
              {message.content}
            </div>
          ) : (
            /* AI messages */
            <div className="max-w-none break-words">
              <ReactMarkdown
                remarkPlugins={[remarkGfm]}
                components={{
                  /* =========================
                     HEADINGS
                     ========================= */

                  h1: ({ children }) => (
                    <h1 className="mb-3 mt-1 text-lg font-semibold tracking-tight">
                      {children}
                    </h1>
                  ),

                  h2: ({ children }) => (
                    <h2 className="mb-2.5 mt-4 text-base font-semibold tracking-tight first:mt-0">
                      {children}
                    </h2>
                  ),

                  h3: ({ children }) => (
                    <h3 className="mb-2 mt-3 text-sm font-semibold first:mt-0">
                      {children}
                    </h3>
                  ),

                  /* =========================
                     PARAGRAPHS
                     ========================= */

                  p: ({ children }) => (
                    <p className="mb-3 last:mb-0">
                      {children}
                    </p>
                  ),

                  /* =========================
                     LISTS
                     ========================= */

                  ul: ({ children }) => (
                    <ul className="mb-3 list-disc space-y-1.5 pl-5">
                      {children}
                    </ul>
                  ),

                  ol: ({ children }) => (
                    <ol className="mb-3 list-decimal space-y-1.5 pl-5">
                      {children}
                    </ol>
                  ),

                  li: ({ children }) => (
                    <li className="pl-1">
                      {children}
                    </li>
                  ),

                  /* =========================
                     TEXT EMPHASIS
                     ========================= */

                  strong: ({ children }) => (
                    <strong className="font-semibold text-foreground">
                      {children}
                    </strong>
                  ),

                  em: ({ children }) => (
                    <em className="italic">
                      {children}
                    </em>
                  ),

                  /* =========================
                     LINKS
                     ========================= */

                  a: ({ children, href }) => (
                    <a
                      href={href}
                      target="_blank"
                      rel="noreferrer"
                      className="font-medium text-primary underline underline-offset-2"
                    >
                      {children}
                    </a>
                  ),

                  /* =========================
                     BLOCKQUOTES
                     ========================= */

                  blockquote: ({ children }) => (
                    <blockquote className="my-3 border-l-2 border-primary/30 pl-3 italic text-muted-foreground">
                      {children}
                    </blockquote>
                  ),

                  /* =========================
                     TABLES
                     ========================= */

                  table: ({ children }) => (
                    <div className="my-4 w-full overflow-x-auto rounded-lg border border-border">
                      <table className="w-full min-w-[520px] border-collapse text-xs sm:text-sm">
                        {children}
                      </table>
                    </div>
                  ),

                  thead: ({ children }) => (
                    <thead className="bg-muted/50">
                      {children}
                    </thead>
                  ),

                  tbody: ({ children }) => (
                    <tbody>
                      {children}
                    </tbody>
                  ),

                  tr: ({ children }) => (
                    <tr className="border-b border-border last:border-b-0">
                      {children}
                    </tr>
                  ),

                  th: ({ children }) => (
                    <th className="whitespace-nowrap border-b border-border px-3 py-2.5 text-left font-semibold">
                      {children}
                    </th>
                  ),

                  td: ({ children }) => (
                    <td className="px-3 py-2.5 align-top">
                      {children}
                    </td>
                  ),

                  /* =========================
                     HORIZONTAL RULE
                     ========================= */

                  hr: () => (
                    <hr className="my-4 border-border" />
                  ),

                  /* =========================
                     CODE
                     ========================= */

                  code: ({ children }) => (
                    <code className="rounded bg-muted px-1.5 py-0.5 text-xs font-medium">
                      {children}
                    </code>
                  ),

                  pre: ({ children }) => (
                    <pre className="my-3 overflow-x-auto rounded-lg bg-muted p-3 text-xs">
                      {children}
                    </pre>
                  ),

                  /* =========================
                     LINE BREAKS
                     ========================= */

                  br: () => <br />,
                }}
              >
                {message.content}
              </ReactMarkdown>
            </div>
          )}
        </div>

        {/* Timestamp */}
        <span
          className={cn(
            "mt-1 block text-[11px] text-muted",
            isUser && "text-right"
          )}
        >
          {message.timestamp}
        </span>
      </div>
    </motion.div>
  );
}

/* =========================================================
   TYPING INDICATOR
   ========================================================= */

export function TypingIndicator() {
  return (
    <div className="flex items-end gap-2.5">
      {/* AI Avatar */}
      <span className="flex h-7 w-7 shrink-0 items-center justify-center rounded-full bg-primary text-white">
        <Activity className="h-3.5 w-3.5" />
      </span>

      {/* Animated dots */}
      <div className="flex items-center gap-1 rounded-2xl rounded-bl-md border border-border bg-white px-4 py-3.5">
        {[0, 1, 2].map((i) => (
          <motion.span
            key={i}
            className="h-1.5 w-1.5 rounded-full bg-muted"
            animate={{
              y: [0, -4, 0],
            }}
            transition={{
              duration: 0.9,
              repeat: Infinity,
              delay: i * 0.15,
            }}
          />
        ))}
      </div>
    </div>
  );
}
