export const ROUTES = {
  HOME: "/",
  TRACK: "/:trackId",
  KATA: "/:phaseId/:kataId",
} as const;

export function trackPath(trackId: string): string {
  return `/${trackId}`;
}

export function kataPath(
  trackId: string,
  phaseId: number,
  kataId: string,
): string {
  return `/${trackId}/${phaseId}/${kataId}`;
}
