# Load JSON file
$json = Get-Content 'synthetic_financial_dataset_20260302.json' -Raw | ConvertFrom-Json

# Create subset with exactly 3 entries per category
$subset = @{}

$json | Get-Member -MemberType NoteProperty | ForEach-Object {
    $key = $_.Name
    $entries = $json.$key
    
    if ($entries -is [System.Object[]]) {
        # Array: take first 3
        $subset[$key] = @($entries[0..2])
    } elseif ($entries -is [System.Collections.Generic.List`1[System.Object]]) {
        # List: convert to array and take first 3
        $subset[$key] = @($entries.ToArray()[0..2])
    } else {
        # Single object or other type
        $subset[$key] = $entries
    }
}

# Save subset with proper formatting
$jsonString = $subset | ConvertTo-Json -Depth 100
$jsonString | Set-Content 'synthetic_financial_dataset_subset.json' -Encoding UTF8

# Print summary
Write-Host "Original dataset:"
$json | Get-Member -MemberType NoteProperty | ForEach-Object {
    $key = $_.Name
    $entries = $json.$key
    if ($entries -is [System.Object[]]) {
        Write-Host "  $key`: $($entries.Count) entries"
    } else {
        Write-Host "  $key`: 1 entry"
    }
}

Write-Host ""
Write-Host "Subset dataset (3 entries per category):"
$subset | Get-Member -MemberType NoteProperty | ForEach-Object {
    $key = $_.Name
    $entries = $subset[$key]
    if ($entries -is [System.Object[]]) {
        Write-Host "  $key`: $($entries.Count) entries"
    } else {
        Write-Host "  $key`: 1 entry"
    }
}

Write-Host ""
Write-Host "Subset saved to: synthetic_financial_dataset_subset.json"
