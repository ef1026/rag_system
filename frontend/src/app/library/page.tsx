import { AppShell } from "@/components/layout/AppShell";
import { FileLibraryPage } from "@/features/file-manager/FileLibraryPage";

export default function LibraryPage() {
  return (
    <AppShell>
      <FileLibraryPage />
    </AppShell>
  );
}
