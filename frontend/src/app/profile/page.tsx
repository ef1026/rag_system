import { AppShell } from "@/components/layout/AppShell";
import { ProfileEditor } from "@/features/profile/ProfileEditor";

export default function ProfilePage() {
  return (
    <AppShell>
      <ProfileEditor />
    </AppShell>
  );
}
