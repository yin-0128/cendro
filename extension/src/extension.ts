import * as vscode from "vscode";

interface ReviewResponse {
  review: string;
  model: string;
  latency_ms: number;
}

const LANGUAGE_MAP: Record<string, string> = {
  typescriptreact: "typescript",
  javascriptreact: "javascript",
  shellscript: "bash",
};

function normalizeLanguage(languageId: string): string {
  return LANGUAGE_MAP[languageId] ?? languageId;
}

async function reviewSelection(): Promise<void> {
  const editor = vscode.window.activeTextEditor;
  if (!editor) {
    vscode.window.showWarningMessage("Cendro: open a file and select code to review.");
    return;
  }

  const selection = editor.selection;
  const code = selection.isEmpty
    ? editor.document.getText()
    : editor.document.getText(selection);

  if (!code.trim()) {
    vscode.window.showWarningMessage("Cendro: nothing to review.");
    return;
  }

  const serverUrl = vscode.workspace
    .getConfiguration("cendro")
    .get<string>("serverUrl", "http://localhost:8000");
  const language = normalizeLanguage(editor.document.languageId);

  await vscode.window.withProgress(
    { location: vscode.ProgressLocation.Notification, title: "Cendro reviewing…" },
    async () => {
      try {
        const res = await fetch(`${serverUrl}/review`, {
          method: "POST",
          headers: { "Content-Type": "application/json" },
          body: JSON.stringify({ code, language }),
        });

        if (!res.ok) {
          const detail = await res.text();
          vscode.window.showErrorMessage(
            `Cendro server returned ${res.status}. Is \`cendro serve\` running? ${detail}`
          );
          return;
        }

        const data = (await res.json()) as ReviewResponse;
        const doc = await vscode.workspace.openTextDocument({
          content:
            `# Cendro review (${data.model}, ${data.latency_ms} ms)\n\n` + data.review,
          language: "markdown",
        });
        await vscode.window.showTextDocument(doc, { preview: true, viewColumn: vscode.ViewColumn.Beside });
      } catch (err) {
        vscode.window.showErrorMessage(
          `Cendro: could not reach ${serverUrl}. Start the server with \`cendro serve\`. (${String(err)})`
        );
      }
    }
  );
}

export function activate(context: vscode.ExtensionContext): void {
  context.subscriptions.push(
    vscode.commands.registerCommand("cendro.review", reviewSelection)
  );
}

export function deactivate(): void {
  // no-op
}
