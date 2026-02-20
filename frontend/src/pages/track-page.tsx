import { createResource, type ParentProps } from "solid-js";
import { useParams } from "@solidjs/router";
import { fetchKatas } from "../lib/api-client";
import { MainLayout } from "../components/layout/main-layout";

export function TrackPage(props: ParentProps) {
  const params = useParams<{ trackId: string }>();
  const [katasData] = createResource(() => params.trackId, fetchKatas);

  return (
    <MainLayout trackId={params.trackId} katasData={katasData}>
      {props.children}
    </MainLayout>
  );
}
