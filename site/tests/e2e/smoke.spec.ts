import { expect, test, type Page } from '@playwright/test'

const slugs = ['game', 'tokens', 'embeddings', 'position', 'attention', 'heads', 'mlp', 'block', 'vote', 'training', 'build', 'family', 'scale']

async function open(page: Page, slug: string): Promise<string[]> {
  const errors: string[] = []
  page.on('console', (m) => m.type() === 'error' && errors.push(m.text()))
  page.on('pageerror', (e) => errors.push(String(e)))
  await page.goto(`/#/${slug}`)
  await page.waitForSelector('article')
  return errors
}

const nextToken = (page: Page) => page.locator('header .tok').innerText()

for (const slug of slugs) {
  test(`${slug}: loads, reacts to a new sentence, fits a phone`, async ({ page }) => {
    const errors = await open(page, slug)
    await expect(page.locator('article h1')).toBeVisible()
    await expect(page.locator('.quiz').first()).toBeVisible()

    await page.fill('#sentence', 'Once upon a time, there was a little girl named')
    await page.waitForTimeout(300)
    await page.fill('#sentence', 'She was very happy because her mom gave her a')
    await page.waitForTimeout(300)

    await page.setViewportSize({ width: 390, height: 800 })
    await page.waitForTimeout(200)
    const overflow = await page.evaluate(() => document.documentElement.scrollWidth - window.innerWidth)
    expect(overflow, 'no sideways scrolling on a phone').toBeLessThanOrEqual(0)
    expect(errors).toEqual([])
  })
}

test('the prediction follows the sentence', async ({ page }) => {
  await open(page, 'game')
  await page.fill('#sentence', 'Once upon a time, there was a little girl named')
  const a = await nextToken(page)
  await page.fill('#sentence', 'The sun was hot and the sky was')
  await expect.poll(() => nextToken(page)).not.toBe(a)
})

test('game: adding the top guess extends the sentence', async ({ page }) => {
  await open(page, 'game')
  const before = await page.inputValue('#sentence')
  await page.getByRole('button', { name: 'Add the top guess' }).click()
  expect((await page.inputValue('#sentence')).length).toBeGreaterThan(before.length)
})

test('game: a round can be played', async ({ page }) => {
  await open(page, 'game')
  await page.fill('#guess', 'garden')
  await page.getByRole('button', { name: 'Reveal' }).click()
  await expect(page.getByText('The story continued with')).toBeVisible()
})

test('tokens: merges can be stepped through', async ({ page }) => {
  await open(page, 'tokens')
  await page.getByRole('button', { name: 'Apply next merge' }).click()
  await expect(page.getByText(/Merge 1 of/)).toBeVisible()
  await page.getByRole('button', { name: 'Merge the most common pair' }).click()
  await expect(page.getByText('Rules learned so far')).toBeVisible()
})

test('heads: silencing a head re-runs the model', async ({ page }) => {
  await open(page, 'heads')
  await page.locator('.sw').first().click()
  await expect(page.getByText('1 silenced')).toBeVisible()
  await page.getByRole('button', { name: 'Test each head on its own' }).click()
  await expect(page.locator('.sw .num').first()).toBeVisible()
})

test('vote: the model writes', async ({ page }) => {
  await open(page, 'vote')
  await page.getByRole('button', { name: 'Write one token' }).click()
  await expect(page.locator('.story mark')).not.toBeEmpty()
})

test('training: an early snapshot can be loaded', async ({ page }) => {
  await open(page, 'training')
  await page.locator('.stepbtn').first().click()
  await expect(page.getByText('The model after 0 training steps')).toBeVisible()
  await expect(page.getByText('loading')).toHaveCount(0)
})

test('family: the three families and the unmasked model', async ({ page }) => {
  await open(page, 'family')
  await page.getByRole('button', { name: 'Encoder + decoder' }).click()
  await expect(page.getByText('3. Decoder: cross-attention')).toBeVisible()
  await page.getByLabel('Causal mask on').uncheck()
  await expect(page.getByText('with the future visible')).toBeVisible()
})

test('scale: the GPT-2 preset gives the published size', async ({ page }) => {
  await open(page, 'scale')
  await page.getByRole('button', { name: 'GPT-2 small (2019)' }).click()
  await expect(page.getByText('124,439,808 weights')).toBeVisible()
})

test('scale: the pipeline can be rebuilt in order', async ({ page }) => {
  await open(page, 'scale')
  const first = page.locator('.pool .card', { hasText: 'Cut the text into tokens' })
  await page.locator('.pool .card', { hasText: 'Softmax turns' }).click()
  await expect(page.getByText('Not yet.')).toBeVisible()
  await first.click()
  await expect(page.locator('.done li')).toHaveCount(1)
})
