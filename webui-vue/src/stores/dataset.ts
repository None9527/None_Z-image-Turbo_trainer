import { defineStore } from 'pinia'
import { ref, computed } from 'vue'
import axios from 'axios'

export interface DatasetImage {
  path: string
  filename: string
  width: number
  height: number
  size: number
  caption?: string
  hasLatentCache: boolean
  hasTextCache: boolean
  thumbnailUrl: string
}

export interface Pagination {
  page: number
  pageSize: number
  totalPages: number
  totalCount: number
  hasNext: boolean
  hasPrev: boolean
}

export interface DatasetInfo {
  path: string
  name: string
  imageCount: number
  totalSize: number
  images: DatasetImage[]
  pagination?: Pagination
  // 全局缓存统计
  totalLatentCached?: number
  totalTextCached?: number
}

export const useDatasetStore = defineStore('dataset', () => {
  const datasets = ref<DatasetInfo[]>([])
  const currentDataset = ref<DatasetInfo | null>(null)
  const isLoading = ref(false)
  const selectedImages = ref<Set<string>>(new Set())

  // 分页状态
  const currentPage = ref(1)
  const pageSize = ref(100)
  const pagination = ref<Pagination | null>(null)

  const currentImages = computed(() => currentDataset.value?.images || [])

  async function scanDataset(path: string, page: number = 1, size: number = 100) {
    isLoading.value = true
    try {
      const response = await axios.post('/api/dataset/scan', {
        path,
        page,
        page_size: size
      })
      const datasetInfo: DatasetInfo = response.data

      // 保存分页信息
      if (response.data.pagination) {
        pagination.value = response.data.pagination
        currentPage.value = response.data.pagination.page
        pageSize.value = response.data.pagination.pageSize
      }

      // 添加或更新数据集列表
      const existingIndex = datasets.value.findIndex(d => d.path === path)
      if (existingIndex >= 0) {
        datasets.value[existingIndex] = datasetInfo
      } else {
        datasets.value.push(datasetInfo)
      }

      // 翻页时保留之前的缓存统计（避免被重置为 null）
      if (currentDataset.value && currentDataset.value.path === path) {
        datasetInfo.totalLatentCached = datasetInfo.totalLatentCached ?? currentDataset.value.totalLatentCached
        datasetInfo.totalTextCached = datasetInfo.totalTextCached ?? currentDataset.value.totalTextCached
      }

      currentDataset.value = datasetInfo
      return datasetInfo
    } catch (error) {
      console.error('Failed to scan dataset:', error)
      throw error
    } finally {
      isLoading.value = false
    }
  }

  // 加载指定页
  async function loadPage(page: number) {
    if (!currentDataset.value) return
    return scanDataset(currentDataset.value.path, page, pageSize.value)
  }

  // 修改每页数量
  async function changePageSize(size: number) {
    pageSize.value = size
    currentPage.value = 1
    if (currentDataset.value) {
      return scanDataset(currentDataset.value.path, 1, size)
    }
  }

  // 异步获取缓存统计（不阻塞首次加载）
  const isLoadingStats = ref(false)

  async function fetchStats(path?: string) {
    const targetPath = path || currentDataset.value?.path
    if (!targetPath) return

    isLoadingStats.value = true
    try {
      const response = await axios.post('/api/dataset/stats', { path: targetPath })
      // 更新当前数据集的缓存统计
      if (currentDataset.value && currentDataset.value.path === targetPath) {
        currentDataset.value.totalLatentCached = response.data.totalLatentCached
        currentDataset.value.totalTextCached = response.data.totalTextCached
      }
      return response.data
    } catch (error) {
      console.error('Failed to fetch stats:', error)
    } finally {
      isLoadingStats.value = false
    }
  }

  async function loadCaption(imagePath: string) {
    try {
      const response = await axios.get(`/api/dataset/caption?path=${encodeURIComponent(imagePath)}`)
      return response.data.caption
    } catch (error) {
      console.error('Failed to load caption:', error)
      return null
    }
  }

  async function saveCaption(imagePath: string, caption: string) {
    try {
      await axios.post('/api/dataset/caption', { path: imagePath, caption })

      // 更新本地状态
      if (currentDataset.value) {
        const image = currentDataset.value.images.find(img => img.path === imagePath)
        if (image) {
          image.caption = caption
        }
      }
      return true
    } catch (error) {
      console.error('Failed to save caption:', error)
      return false
    }
  }

  async function generateCaptions(modelType: 'qwen' | 'blip' = 'qwen') {
    try {
      if (!currentDataset.value) return

      const response = await axios.post('/api/dataset/generate-captions', {
        datasetPath: currentDataset.value.path,
        modelType
      })
      return response.data
    } catch (error) {
      console.error('Failed to generate captions:', error)
      throw error
    }
  }

  function toggleImageSelection(imagePath: string) {
    if (selectedImages.value.has(imagePath)) {
      selectedImages.value.delete(imagePath)
    } else {
      selectedImages.value.add(imagePath)
    }
  }

  function selectAll() {
    if (currentDataset.value) {
      currentDataset.value.images.forEach(img => {
        selectedImages.value.add(img.path)
      })
    }
  }

  function clearSelection() {
    selectedImages.value.clear()
  }

  return {
    datasets,
    currentDataset,
    currentImages,
    isLoading,
    selectedImages,
    // 分页相关
    currentPage,
    pageSize,
    pagination,
    // 缓存统计相关
    isLoadingStats,
    fetchStats,
    // 方法
    scanDataset,
    loadPage,
    changePageSize,
    loadCaption,
    saveCaption,
    generateCaptions,
    toggleImageSelection,
    selectAll,
    clearSelection
  }
})

