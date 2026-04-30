from pathlib import Path


def rename_wav_files(directory: Path) -> None:
    wav_files = sorted(directory.glob('*.wav'), key=lambda p: p.name.lower())
    if not wav_files:
        print('No .wav files found in', directory)
        return

    targets = [directory / f'krage_{i}{path.suffix.lower()}' for i, path in enumerate(wav_files, start=1)]

    existing_conflicts = [target for target in targets if target.exists() and target not in wav_files]
    if existing_conflicts:
        print('Cannot rename because these target files already exist:')
        for conflict in existing_conflicts:
            print('  ', conflict.name)
        return

    temp_paths = []
    for original in wav_files:
        temp = original.with_name(original.name + '.tmp-renaming')
        if temp.exists():
            raise FileExistsError(f'Temporary file already exists: {temp}')
        original.rename(temp)
        temp_paths.append(temp)

    for temp_path, target in zip(temp_paths, targets):
        temp_path.rename(target)

    print(f'Renamed {len(wav_files)} .wav files to krage_1..krage_{len(wav_files)}')


if __name__ == '__main__':
    rename_wav_files(Path(__file__).parent)
