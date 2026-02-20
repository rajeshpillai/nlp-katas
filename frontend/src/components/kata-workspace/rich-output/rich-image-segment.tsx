export function RichImageSegment(props: { content: string }) {
  return (
    <div class="rich-output-image">
      <img src={props.content} alt="Plot output" class="rich-output-image-img" />
    </div>
  );
}
