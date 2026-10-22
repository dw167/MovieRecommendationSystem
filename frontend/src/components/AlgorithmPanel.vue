<script setup lang="ts">
import { computed, onScopeDispose, ref, watch } from 'vue'
import { Connection } from '@element-plus/icons-vue'
import http, { ApiError } from '../api'
import StatePanel from './StatePanel.vue'
import MovieCard from './MovieCard.vue'
import type { AlgorithmResult, Movie } from '../types'
const props = withDefaults(defineProps<{ endpoint: string | null; label: string; emptySubject?: string }>(), { emptySubject: '登录后体验' })
const loading = ref(false), error = ref<unknown>(null), unavailable = ref(false)
const result = ref<AlgorithmResult | null>(null)
const movies = ref<Movie[]>([])
let run = 0
onScopeDispose(() => { run++ })
async function load() {
  const seq = ++run
  error.value = null; result.value = null; unavailable.value = false; loading.value = false; movies.value = []
  if (!props.endpoint) return
  loading.value = true
  try {
    const response = await http.get<AlgorithmResult>(props.endpoint)
    if (seq !== run) return
    result.value = response.data
    if (response.data.items.length) {
      const details = await Promise.all(
        response.data.items.map(item => http.get<Movie>(`/movies/${item.movie_id}`).then(r => r.data).catch(() => null))
      )
      if (seq !== run) return
      movies.value = details.filter((m): m is Movie => m !== null)
    }
  } catch (e) {
    if (seq !== run) return
    if (e instanceof ApiError && e.status === 501 && e.code === 'ALGORITHM_NOT_IMPLEMENTED') unavailable.value = true
    else error.value = e
  } finally { if (seq === run) loading.value = false }
}
watch(() => props.endpoint, load, { immediate: true })
const strategy = computed(() => {
  if (!result.value || result.value.algorithm !== 'cold_start' || !result.value.items.length) return null
  const source = result.value.items[0].recall_sources?.[0]
  if (source === 'popular-fallback') return { text: '策略：热门榜单兜底 · 非个性化', hint: '该用户评分数 < 3，尚不能构建兴趣画像，展示全体用户加权热门榜单。' }
  if (source === 'content-personalized') return { text: '策略：内容个性化推荐', hint: '该用户已有 ≥3 条评分，基于其历史正反馈的类型画像做个性化推荐。' }
  return null
})
</script>
<template>
  <section class="algorithm-panel" :aria-label="label">
    <div class="section-top"><h3>{{ label }}</h3><span class="small-label">RECOMMENDATION</span></div>
    <StatePanel :loading="loading" :error="error" @retry="load">
      <div v-if="!endpoint || unavailable" class="algorithm-empty" role="status">
        <div class="algorithm-symbol"><el-icon aria-hidden="true"><Connection /></el-icon></div>
        <div><span class="status-pill">{{ !endpoint ? emptySubject : '尚未实现' }}</span><h3>{{ !endpoint ? '选择观众，开始探索' : '下一部好电影，等待算法发现' }}</h3><p>{{ !endpoint ? '先选择历史用户或登录注册账号。推荐功能等待算法接入。' : '等待算法接入；当前没有模型版本或推荐结果。' }}</p></div>
        <RouterLink v-if="!endpoint && emptySubject === '登录后体验'" class="text-link" to="/login">前往登录 →</RouterLink>
      </div>
      <div v-else-if="result" class="recommendation-results">
        <div class="result-meta">
          <span class="status-pill">{{ result.status }}</span>
          <span v-if="strategy" class="strategy-pill">{{ strategy.text }}</span>
          <span class="muted">模型 {{ result.model_version }} · 数据集 {{ result.dataset_version }}</span>
        </div>
        <p v-if="strategy" class="strategy-hint">{{ strategy.hint }}</p>
        <p v-if="!result.items.length" class="muted">本次没有推荐结果。</p>
        <div v-else class="recommendation-grid">
          <div v-for="(item, idx) in result.items" :key="item.movie_id" class="recommendation-item">
            <MovieCard v-if="movies[idx]" :movie="movies[idx]" />
            <div v-else class="card-fallback">电影 #{{ item.movie_id }}</div>
            <div class="recommendation-badge">#{{ item.rank }}<span v-if="item.score !== null"> · {{ item.score.toFixed(4) }}</span></div>
          </div>
        </div>
      </div>
    </StatePanel>
  </section>
</template>

<style scoped>
.recommendation-results { margin-top: 8px; }
.result-meta { display: flex; align-items: center; gap: 12px; margin-bottom: 16px; flex-wrap: wrap; }
.strategy-pill {
  background: rgba(212, 175, 102, 0.12); color: var(--accent);
  font-size: 12px; font-weight: 600; padding: 3px 10px;
  border-radius: 999px; border: 1px solid var(--border);
}
.strategy-hint { margin: -8px 0 16px; color: var(--muted); font-size: 12px; }
.recommendation-grid { display: grid; grid-template-columns: repeat(auto-fill, minmax(160px, 1fr)); gap: 20px; }
.recommendation-item { position: relative; }
.recommendation-badge {
  position: absolute; top: 8px; left: 8px; z-index: 2;
  background: rgba(10, 18, 19, 0.85); backdrop-filter: blur(4px);
  color: var(--accent); font-size: 12px; font-weight: 700;
  padding: 4px 8px; border-radius: 6px; border: 1px solid var(--border);
}
.card-fallback { aspect-ratio: 2/3; background: var(--surface); border: 1px solid var(--border); border-radius: 8px; display: flex; align-items: center; justify-content: center; color: var(--muted); font-size: 13px; }
</style>
