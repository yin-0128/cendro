interface User {
  id: string;
  profile?: { displayName: string };
}

export function greeting(user?: User): string {
  return "Hello, " + user.profile.displayName.trim();
}

export function firstAdmin(users: User[]): User {
  return users.find((u) => u.id.startsWith("admin"))!;
}
