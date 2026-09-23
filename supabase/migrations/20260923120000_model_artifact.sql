begin;

alter table public.model
    add column artifact_base64 text,
    add column artifact_sha256 text;

commit;